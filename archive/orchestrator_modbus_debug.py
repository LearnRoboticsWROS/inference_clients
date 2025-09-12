#!/usr/bin/env python3
"""
Orchestratore Modbus DEBUG (Jetson = Modbus Server)
- Espone Modbus TCP su 0.0.0.0:5020
- Mappa:
  Coils (PLC->Jetson): C0 START_CAPTURE, C1 ACK_RESET
  Discrete Inputs (Jetson->PLC): DI0 BUSY, DI1 RESULT_READY, DI2 ERROR
  Holding Registers (Jetson->PLC): HR0 LIP_RESULT, HR1 BODY_RESULT, HR2 ERROR_CODE

- Quando START_CAPTURE=1:
    * Jetson resetta START_CAPTURE=0
    * DI0 BUSY=1
    * lancia lip_client.py e body_client.py in parallelo
    * aggiorna HR0/HR1/HR2 + DI1 RESULT_READY=1 + DI0 BUSY=0
    * attende ACK_RESET=1, poi resetta e torna idle

- DEBUG: stampa ogni write ricevuto dal PLC e ogni update fatto dalla Jetson
- Tastiera:
    '1' -> simula START_CAPTURE
    'r' -> simula ACK_RESET
    'q' -> quit
"""

import sys
import time
import threading
import argparse
import subprocess
from queue import Queue, Empty
from typing import Optional, Tuple, List

# ====== PATH CLIENT INFERENZA (adatta ai tuoi path) ======
PY_INFER   = "/home/bewater/big400_bottle_inspection/inference_clients/.venv/bin/python"
LIP_SCRIPT = "/home/bewater/big400_bottle_inspection/inference_clients/lip_client.py"
BODY_SCRIPT= "/home/bewater/big400_bottle_inspection/inference_clients/body_client.py"

LIP_EXTRA_ARGS  = ["--cam-index","0"]   # cambia se serve
BODY_EXTRA_ARGS = ["--cam-index","1"]

# timeout robusto (finestra 6s + margine)
CLIENT_TIMEOUT_S = 15.0
ACK_TIMEOUT_S    = 10.0

# ====== MODBUS ======
MODBUS_BIND_IP   = "0.0.0.0"
MODBUS_BIND_PORT = 5020

# Adresses
COIL_START = 0
COIL_ACK   = 1

DI_BUSY    = 0
DI_READY   = 1
DI_ERROR   = 2

HR_LIP     = 0
HR_BODY    = 1
HR_ERR     = 2

# Error codes
ERR_NONE   = 0
ERR_TIMEOUT= 2
ERR_CLIENT = 3
ERR_INFER  = 4

# ====== pymodbus 2.5.3 ======
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock

# ---------- DataBlock con logging ----------
class LoggingDataBlock(ModbusSequentialDataBlock):
    """DataBlock che logga ogni write del client (PLC)."""
    def __init__(self, address: int, values: List[int], name: str):
        super().__init__(address, values)
        self.name = name
        self.on_write = None  # callback(name, address, values)

    def setValues(self, address, values):
        super().setValues(address, values)
        if self.on_write:
            try:
                self.on_write(self.name, address, list(values))
            except Exception as e:
                print(f"[DataBlock][{self.name}] on_write error: {e}", file=sys.stderr)

# ---------- Helpers Modbus ----------
def set_coil(ctx: ModbusServerContext, idx: int, val: int):
    ctx[0x00].setValues(1, idx, [1 if val else 0])

def get_coil(ctx: ModbusServerContext, idx: int, count: int = 1) -> List[int]:
    return ctx[0x00].getValues(1, idx, count)

def set_di(ctx: ModbusServerContext, idx: int, val: int):
    ctx[0x00].setValues(2, idx, [1 if val else 0])

def get_di(ctx: ModbusServerContext, idx: int, count: int = 1) -> List[int]:
    return ctx[0x00].getValues(2, idx, count)

def set_hr(ctx: ModbusServerContext, idx: int, val: int):
    ctx[0x00].setValues(3, idx, [int(val)])

def get_hr(ctx: ModbusServerContext, idx: int, count: int = 1) -> List[int]:
    return ctx[0x00].getValues(3, idx, count)

# ---------- Server thread ----------
def start_modbus_server(context: ModbusServerContext, bind_ip: str, bind_port: int):
    t = threading.Thread(
        target=StartTcpServer,
        kwargs={"context": context, "address": (bind_ip, bind_port)},
        daemon=True
    )
    t.start()
    return t

# ---------- Keyboard (simulazione PLC da terminale) ----------
def keyboard_thread(q: Queue):
    print(">> Tastiera: '1' = START_CAPTURE, 'r' = ACK_RESET, 'q' = quit")
    while True:
        try:
            s = input().strip().lower()
        except EOFError:
            break
        if s in ("1","r","q"):
            q.put(s)
        else:
            print("  (comando sconosciuto, usa 1/r/q)")

# ---------- Run client inferenza ----------
def run_client(cmd: list, expected_prefix: str, timeout_s: float) -> Tuple[Optional[str], str, int]:
    """
    Esegue un client e ritorna: (stdout_token, stderr, returncode)
    stdout_token ∈ {LIP_OK, LIP_NOK, BODY_OK, BODY_NOK} oppure None.
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, f"[{expected_prefix}] TIMEOUT > {timeout_s}s", 124
    except Exception as e:
        return None, f"[{expected_prefix}] Exception: {e}", 1

    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    rc  = p.returncode

    out_up = out.upper()
    ok_tok  = f"{expected_prefix}_OK"
    nok_tok = f"{expected_prefix}_NOK"
    if out_up == ok_tok or out_up == nok_tok:
        return out_up, err, rc

    if rc != 0:
        if not err:
            err = f"[{expected_prefix}] rc={rc} output inatteso: '{out}'"
        return None, err, rc
    else:
        return None, f"[{expected_prefix}] output inatteso rc=0: '{out}'", rc

# ---------- Orchestrazione ----------
def do_inference_cycle(ctx: ModbusServerContext, args):
    # DEBUG reset stato / risultati
    print("[ORCH] START_CAPTURE ricevuto -> avvio ciclo inferenza")

    # Reset campi prima di iniziare
    set_di(ctx, DI_ERROR, 0)
    set_di(ctx, DI_READY, 0)
    set_hr(ctx, HR_LIP,  0)
    set_hr(ctx, HR_BODY, 0)
    set_hr(ctx, HR_ERR,  ERR_NONE)

    # Jetson prende in carico: BUSY=1 e reset START
    set_di(ctx, DI_BUSY, 1)
    set_coil(ctx, COIL_START, 0)
    print(f"[MODBUS->PLC] DI_BUSY=1, DI_READY=0, HR(LIP,BODY,ERR)=(0,0,0), reset COIL_START=0")

    # prepara comandi
    lip_cmd  = [args.py_infer, args.lip_script]  + args.lip_args
    body_cmd = [args.py_infer, args.body_script] + args.body_args

    # esegui in parallelo
    results = {"LIP": None, "BODY": None}
    errors  = {"LIP": "",   "BODY": ""}
    rcodes  = {"LIP": 0,    "BODY": 0}

    def run_lip():
        res, err, rc = run_client(lip_cmd, "LIP", args.timeout)
        results["LIP"] = res; errors["LIP"] = err; rcodes["LIP"] = rc

    def run_body():
        res, err, rc = run_client(body_cmd, "BODY", args.timeout)
        results["BODY"] = res; errors["BODY"] = err; rcodes["BODY"] = rc

    tl = threading.Thread(target=run_lip,  daemon=True)
    tb = threading.Thread(target=run_body, daemon=True)
    t0 = time.time()
    tl.start(); tb.start()
    tl.join();  tb.join()
    elapsed = time.time() - t0

    # elabora risultati
    lip_out  = results["LIP"]  or "LIP_ERROR"
    body_out = results["BODY"] or "BODY_ERROR"
    print(f"[ORCH] lip={lip_out}, body={body_out}, elapsed={elapsed:.2f}s")
    if errors["LIP"]:
        print(f"[LIP][stderr] {errors['LIP']}", file=sys.stderr)
    if errors["BODY"]:
        print(f"[BODY][stderr] {errors['BODY']}", file=sys.stderr)

    # set HR in base ai token
    def token_to_hr(tok: str, prefix: str) -> int:
        if tok == f"{prefix}_OK":  return 1
        if tok == f"{prefix}_NOK": return 2
        return 0

    lip_hr  = token_to_hr(lip_out,  "LIP")
    body_hr = token_to_hr(body_out, "BODY")
    set_hr(ctx, HR_LIP,  lip_hr)
    set_hr(ctx, HR_BODY, body_hr)

    # error handling
    err_code = ERR_NONE
    if lip_hr == 0 or body_hr == 0:
        # se uno dei due non ha prodotto token valido, metti un errore
        if rcodes["LIP"] == 124 or rcodes["BODY"] == 124:
            err_code = ERR_TIMEOUT
        elif rcodes["LIP"] != 0 or rcodes["BODY"] != 0:
            err_code = ERR_CLIENT
        else:
            err_code = ERR_INFER
    set_hr(ctx, HR_ERR, err_code)
    set_di(ctx, DI_BUSY, 0)
    set_di(ctx, DI_READY, 1)
    set_di(ctx, DI_ERROR, 1 if err_code != ERR_NONE else 0)

    print(f"[MODBUS->PLC] HR_LIP={lip_hr}, HR_BODY={body_hr}, HR_ERR={err_code}, DI_BUSY=0, DI_READY=1, DI_ERROR={(1 if err_code else 0)}")

    # attende ACK_RESET
    print("[ORCH] In attesa di ACK_RESET (coil 1=1) oppure 'r' da tastiera...")
    t_start_ack = time.time()
    while True:
        # controlla coil ack
        c1 = get_coil(ctx, COIL_ACK, 1)[0]
        if c1 == 1:
            print("[ORCH] ACK_RESET ricevuto dal PLC.")
            break
        # timeout (non strettissimo, solo per debug)
        if (time.time() - t_start_ack) > ACK_TIMEOUT_S:
            print("[ORCH] ACK_RESET timeout, proseguo comunque (debug).")
            break
        time.sleep(0.05)

    # reset stato post-ack
    set_coil(ctx, COIL_ACK, 0)
    set_di(ctx, DI_READY, 0)
    set_hr(ctx, HR_LIP,  0)
    set_hr(ctx, HR_BODY, 0)
    set_hr(ctx, HR_ERR,  ERR_NONE)
    set_di(ctx, DI_ERROR, 0)
    print(f"[MODBUS->PLC] Reset post-ACK: COIL_ACK=0, DI_READY=0, HR_*=0, DI_ERROR=0")

def main():
    ap = argparse.ArgumentParser(description="Orchestratore Modbus DEBUG (Jetson=Server)")
    ap.add_argument("--bind-ip", default=MODBUS_BIND_IP, help="IP bind Modbus server (0.0.0.0)")
    ap.add_argument("--bind-port", type=int, default=MODBUS_BIND_PORT, help="Porta Modbus server (5020)")
    ap.add_argument("--py-infer", default=PY_INFER, help="Path python venv inferenza")
    ap.add_argument("--lip-script", default=LIP_SCRIPT, help="Path lip_client.py")
    ap.add_argument("--body-script", default=BODY_SCRIPT, help="Path body_client.py")
    ap.add_argument("--timeout", type=float, default=CLIENT_TIMEOUT_S, help="Timeout client inferenza (s)")
    ap.add_argument("--lip-args",  default=",".join(LIP_EXTRA_ARGS),  help="CSV argomenti extra lip (es: --cam-index,0)")
    ap.add_argument("--body-args", default=",".join(BODY_EXTRA_ARGS), help="CSV argomenti extra body (es: --cam-index,1)")
    args = ap.parse_args()

    # Costruisci liste argomenti per i client
    args.lip_args  = [s for s in (args.lip_args.split(",") if args.lip_args else []) if s]
    args.body_args = [s for s in (args.body_args.split(",") if args.body_args else []) if s]

    # Datastore con logging
    di_block   = LoggingDataBlock(0, [0]*8,  "DI")
    coil_block = LoggingDataBlock(0, [0]*8,  "COIL")
    hr_block   = LoggingDataBlock(0, [0]*16, "HR")
    ir_block   = LoggingDataBlock(0, [0]*16, "IR")  # non usato, ma presente

    def on_write(name, address, values):
        print(f"[PLC->MODBUS] WRITE {name} addr={address} values={values}")

    coil_block.on_write = on_write
    hr_block.on_write   = on_write
    di_block.on_write   = on_write  # raro: normalmente DI non li scrive il PLC

    store = ModbusSlaveContext(
        di=di_block, co=coil_block, hr=hr_block, ir=ir_block
    )
    context = ModbusServerContext(slaves=store, single=True)

    # inizializza stato
    set_di(context, DI_BUSY, 0)
    set_di(context, DI_READY, 0)
    set_di(context, DI_ERROR, 0)
    set_hr(context, HR_LIP,  0)
    set_hr(context, HR_BODY, 0)
    set_hr(context, HR_ERR,  ERR_NONE)
    set_coil(context, COIL_START, 0)
    set_coil(context, COIL_ACK,   0)

    # avvia server in background
    start_modbus_server(context, args.bind_ip, args.bind_port)
    print(f"[SERVER] Modbus TCP in ascolto su {args.bind_ip}:{args.bind_port}")
    print("         Mappa: Coils[0=START,1=ACK]  DI[0=BUSY,1=READY,2=ERROR]  HR[0=LIP,1=BODY,2=ERR]")

    # thread tastiera (simulazione PLC)
    q = Queue()
    tk = threading.Thread(target=keyboard_thread, args=(q,), daemon=True)
    tk.start()

    print("[ORCH] Pronto. Attendere write PLC su COIL0 oppure premere '1' da tastiera...")
    try:
        while True:
            # 1) evento tastiera
            try:
                cmd = q.get_nowait()
                if cmd == "q":
                    print("[ORCH] Quit richiesto.")
                    break
                elif cmd == "1":
                    print("[SIM] Tastiera -> START_CAPTURE")
                    set_coil(context, COIL_START, 1)
                elif cmd == "r":
                    print("[SIM] Tastiera -> ACK_RESET")
                    set_coil(context, COIL_ACK, 1)
            except Empty:
                pass

            # 2) poll coil START dal PLC
            c0 = get_coil(context, COIL_START, 1)[0]
            if c0 == 1:
                do_inference_cycle(context, args)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[ORCH] CTRL-C -> uscita.")
    finally:
        print("[ORCH] Terminato.")

if __name__ == "__main__":
    main()

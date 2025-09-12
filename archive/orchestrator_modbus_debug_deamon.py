#!/usr/bin/env python3
"""
Orchestratore Modbus DEBUG (Jetson = Modbus Server) - DAEMON CLIENTS
- Espone Modbus TCP su 0.0.0.0:5020
- Mappa:
  Coils (PLC->Jetson): C0 START_CAPTURE, C1 ACK_RESET
  Discrete Inputs (Jetson->PLC): DI0 BUSY, DI1 RESULT_READY, DI2 ERROR
  Holding Registers (Jetson->PLC): HR0 LIP_RESULT, HR1 BODY_RESULT, HR2 ERROR_CODE
    LIP/BODY_RESULT: 0=UNK, 1=OK, 2=NOK
    ERROR_CODE: 0=NONE, 2=TIMEOUT, 3=CLIENT_FAIL, 4=INFERENCE_FAIL

- Flusso (DAEMON):
    * I client di visione (lip/body) sono già avviati con --daemon e finestre aperte
    * Quando START_CAPTURE=1:
        - Jetson resetta START_CAPTURE=0
        - DI0 BUSY=1
        - crea i trigger-file per LIP/BODY
        - attende i result-file per entrambi (o timeout)
        - aggiorna HR0/HR1/HR2 + DI1 READY + DI0 BUSY
        - attende ACK_RESET=1 (o timeout), poi resetta e torna IDLE

- DEBUG: stampa ogni write ricevuto dal PLC e ogni update fatto dalla Jetson
- Tastiera:
    '1' -> simula START_CAPTURE
    'r' -> simula ACK_RESET
    'q' -> quit
"""

import sys
import os
import time
import threading
import argparse
from queue import Queue, Empty
from typing import List

# ====== TIMEOUTS ======
CLIENT_TIMEOUT_S = 15.0   # attesa risultati (6s + margine)
ACK_TIMEOUT_S    = 10.0   # attesa ACK dal PLC

# ====== MODBUS ======
MODBUS_BIND_IP   = "0.0.0.0"
MODBUS_BIND_PORT = 5020

# Indirizzi (0-based)
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

# ====== Trigger/Result files (devono combaciare con i client daemon) ======
LIP_TRIGGER_DEFAULT  = "/tmp/lip_go"
LIP_RESULT_DEFAULT   = "/tmp/lip_res"
BODY_TRIGGER_DEFAULT = "/tmp/body_go"
BODY_RESULT_DEFAULT  = "/tmp/body_res"

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

# ---------- Keyboard (simulazione PLC) ----------
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

# ---------- Trigger/Result helpers ----------
def touch(path: str):
    with open(path, "w") as f:
        f.write("go\n")

def wait_result(path: str, timeout_s: float) -> str:
    """Attende la comparsa del file risultato e ritorna la prima riga (stripped), '' su timeout."""
    t0 = time.time()
    while (time.time() - t0) <= timeout_s:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    line = (f.readline() or "").strip()
                return line
            except:
                pass
        time.sleep(0.05)
    return ""

def cleanup_results(paths: List[str]):
    for p in paths:
        try: os.remove(p)
        except: pass

# ---------- Orchestrazione ----------
def do_inference_cycle(ctx: ModbusServerContext, args):
    print("[ORCH] START_CAPTURE ricevuto -> avvio ciclo inferenza (daemon)")

    # Reset campi prima di iniziare
    set_di(ctx, DI_ERROR, 0)
    set_di(ctx, DI_READY, 0)
    set_hr(ctx, HR_LIP,  0)
    set_hr(ctx, HR_BODY, 0)
    set_hr(ctx, HR_ERR,  ERR_NONE)

    # Jetson BUSY e consumo del trigger
    set_di(ctx, DI_BUSY, 1)
    set_coil(ctx, COIL_START, 0)
    print(f"[MODBUS->PLC] DI_BUSY=1, DI_READY=0, HR(LIP,BODY,ERR)=(0,0,0), reset COIL_START=0")

    # pulizia vecchi risultati
    cleanup_results([args.lip_result, args.body_result])

    # attiva i due client residenti (creo i trigger)
    try:
        touch(args.lip_trigger)
        print(f"[ORCH] trigger LIP -> {args.lip_trigger}")
    except Exception as e:
        print(f"[ORCH] lip trigger error: {e}", file=sys.stderr)

    try:
        touch(args.body_trigger)
        print(f"[ORCH] trigger BODY -> {args.body_trigger}")
    except Exception as e:
        print(f"[ORCH] body trigger error: {e}", file=sys.stderr)

    # attendo risultati
    t0 = time.time()
    lip_tok  = wait_result(args.lip_result,  args.timeout)
    body_tok = wait_result(args.body_result, args.timeout)
    elapsed = time.time() - t0

    if not lip_tok:  lip_tok  = "LIP_ERROR"
    if not body_tok: body_tok = "BODY_ERROR"
    print(f"[ORCH] results: {lip_tok}, {body_tok}  elapsed={elapsed:.2f}s")

    def tok_to_hr(tok: str, prefix: str) -> int:
        if tok == f"{prefix}_OK":  return 1
        if tok == f"{prefix}_NOK": return 2
        return 0

    lip_hr  = tok_to_hr(lip_tok,  "LIP")
    body_hr = tok_to_hr(body_tok, "BODY")
    set_hr(ctx, HR_LIP,  lip_hr)
    set_hr(ctx, HR_BODY, body_hr)

    # error code
    err_code = ERR_NONE
    if lip_hr == 0 or body_hr == 0:
        # se un token manca/è invalido -> timeout o inferenza
        if (lip_tok == "LIP_ERROR" and not os.path.exists(args.lip_result)) or \
           (body_tok == "BODY_ERROR" and not os.path.exists(args.body_result)):
            err_code = ERR_TIMEOUT
        else:
            err_code = ERR_INFER
    set_hr(ctx, HR_ERR, err_code)

    # fine ciclo
    set_di(ctx, DI_BUSY, 0)
    set_di(ctx, DI_READY, 1)
    set_di(ctx, DI_ERROR, 1 if err_code != ERR_NONE else 0)
    print(f"[MODBUS->PLC] HR_LIP={lip_hr}, HR_BODY={body_hr}, HR_ERR={err_code}, DI_BUSY=0, DI_READY=1, DI_ERROR={(1 if err_code else 0)}")

    # attesa ACK
    print("[ORCH] In attesa di ACK_RESET (coil 1=1) oppure 'r' da tastiera...")
    t_start_ack = time.time()
    while True:
        if get_coil(ctx, COIL_ACK, 1)[0] == 1:
            print("[ORCH] ACK_RESET ricevuto dal PLC.")
            break
        if (time.time() - t_start_ack) > args.ack_timeout:
            print("[ORCH] ACK_RESET timeout, proseguo comunque (debug).")
            break
        time.sleep(0.05)

    # reset per prossimo ciclo
    set_coil(ctx, COIL_ACK, 0)
    set_di(ctx, DI_READY, 0)
    set_hr(ctx, HR_LIP,  0)
    set_hr(ctx, HR_BODY, 0)
    set_hr(ctx, HR_ERR,  ERR_NONE)
    set_di(ctx, DI_ERROR, 0)
    cleanup_results([args.lip_result, args.body_result])
    print(f"[MODBUS->PLC] Reset post-ACK: COIL_ACK=0, DI_READY=0, HR_*=0, DI_ERROR=0")

def main():
    ap = argparse.ArgumentParser(description="Orchestratore Modbus DEBUG (daemon clients)")
    ap.add_argument("--bind-ip", default=MODBUS_BIND_IP, help="IP bind Modbus server (0.0.0.0)")
    ap.add_argument("--bind-port", type=int, default=MODBUS_BIND_PORT, help="Porta Modbus server (5020)")
    ap.add_argument("--timeout", type=float, default=CLIENT_TIMEOUT_S, help="Timeout attesa risultati (s)")
    ap.add_argument("--ack-timeout", type=float, default=ACK_TIMEOUT_S, help="Timeout attesa ACK (s)")

    # percorsi trigger/result (devono combaciare con quelli dei client)
    ap.add_argument("--lip-trigger",  default=LIP_TRIGGER_DEFAULT)
    ap.add_argument("--lip-result",   default=LIP_RESULT_DEFAULT)
    ap.add_argument("--body-trigger", default=BODY_TRIGGER_DEFAULT)
    ap.add_argument("--body-result",  default=BODY_RESULT_DEFAULT)
    args = ap.parse_args()

    # Datastore con logging
    di_block   = LoggingDataBlock(0, [0]*8,  "DI")
    coil_block = LoggingDataBlock(0, [0]*8,  "COIL")
    hr_block   = LoggingDataBlock(0, [0]*16, "HR")
    ir_block   = LoggingDataBlock(0, [0]*16, "IR")  # non usato

    def on_write(name, address, values):
        print(f"[PLC->MODBUS] WRITE {name} addr={address} values={values}")

    coil_block.on_write = on_write
    hr_block.on_write   = on_write
    di_block.on_write   = on_write  # raro: normalmente DI non li scrive il PLC

    store = ModbusSlaveContext(di=di_block, co=coil_block, hr=hr_block, ir=ir_block)
    context = ModbusServerContext(slaves=store, single=True)

    # stato iniziale
    set_di(context, DI_BUSY, 0)
    set_di(context, DI_READY, 0)
    set_di(context, DI_ERROR, 0)
    set_hr(context, HR_LIP,  0)
    set_hr(context, HR_BODY, 0)
    set_hr(context, HR_ERR,  ERR_NONE)
    set_coil(context, COIL_START, 0)
    set_coil(context, COIL_ACK,   0)

    # avvia server Modbus
    start_modbus_server(context, args.bind_ip, args.bind_port)
    print(f"[SERVER] Modbus TCP in ascolto su {args.bind_ip}:{args.bind_port}")
    print("         Mappa: Coils[0=START,1=ACK]  DI[0=BUSY,1=READY,2=ERROR]  HR[0=LIP,1=BODY,2=ERR]")
    print("         Modalità DAEMON: client visione già avviati con trigger/result files.")
    print("[ORCH] Pronto. Attendere write PLC su COIL0 oppure premere '1' da tastiera...")

    # thread tastiera (simulazione PLC)
    q = Queue()
    tk = threading.Thread(target=keyboard_thread, args=(q,), daemon=True)
    tk.start()

    try:
        while True:
            # tastiera
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

            # trigger da PLC
            if get_coil(context, COIL_START, 1)[0] == 1:
                do_inference_cycle(context, args)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[ORCH] CTRL-C -> uscita.")
    finally:
        print("[ORCH] Terminato.")

if __name__ == "__main__":
    main()

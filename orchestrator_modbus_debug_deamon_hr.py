#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestratore Modbus DEBUG (DAEMON) - TUTTO SU HOLDING REGISTERS (HR)
Jetson = Modbus TCP Server (default 0.0.0.0:5020)
PLC    = Modbus TCP Client

MAPPING (human addressing -> internal 0-based index):
  40001 (0) START_CAPTURE   : 0/1   (PLC -> Jetson)
  40002 (1) ACK_RESET       : 0/1   (PLC -> Jetson)
  40003 (2) BUSY            : 0/1   (Jetson -> PLC)
  40004 (3) RESULT_READY    : 0/1   (Jetson -> PLC)
  40005 (4) ERROR           : 0/1   (Jetson -> PLC)
  40006 (5) LIP_RESULT      : 0=UNK, 1=OK, 2=NOK
  40007 (6) BODY_RESULT     : 0=UNK, 1=OK, 2=NOK
  40008 (7) BODY_CL_RESULT  : 0=UNK, 1=OK, 2=NOK
  40009 (8) ERROR_CODE      : 0=NONE, 2=TIMEOUT, 3=CLIENT_FAIL, 4=INFERENCE_FAIL

Flow (daemon):
- I client di visione sono già avviati con --daemon e finestre aperte.
- PLC scrive HR[0]=1 (40001 START_CAPTURE): parte un ciclo
- Orchestratore:
    * Consuma START (HR[0]=0)
    * HR[2]=BUSY=1, HR[3]=READY=0, HR[4]=ERROR=0
    * Resetta HR risultati a 0
    * Crea i trigger-file per LIP, BODY, BODY_CL (se configurato)
    * Attende i file risultato (entro timeout) IN PARALLELO
    * Pubblica HR risultati e HR[3]=READY=1, HR[2]=BUSY=0 (+HR[4]/HR[8] per errori)
    * Attende HR[1]=ACK_RESET=1 (o timeout), poi resetta stato e torna IDLE.

Tastiera (debug senza PLC):
  '1' -> simula START_CAPTURE (HR[0]=1)
  'r' -> simula ACK_RESET (HR[1]=1)
  'q' -> quit
"""

import os, sys, time, threading, argparse
from queue import Queue, Empty
from typing import List, Dict, Tuple

# ====== TIMEOUTS (consigli: timeout=8s, ack-timeout=4s con ciclo 9s) ======
RESULT_TIMEOUT_S = 15.0
ACK_TIMEOUT_S    = 10.0

# ====== MODBUS ======
BIND_IP   = "0.0.0.0"
BIND_PORT = 5020

# ====== HR INDICI (0-based) ======
HR_START       = 0   # 40001
HR_ACK         = 1   # 40002
HR_BUSY        = 2   # 40003
HR_READY       = 3   # 40004
HR_ERROR       = 4   # 40005
HR_LIP         = 5   # 40006
HR_BODY        = 6   # 40007
HR_BODY_CL     = 7   # 40008
HR_ERR_CODE    = 8   # 40009

# ====== ERROR CODES ======
ERR_NONE    = 0
ERR_TIMEOUT = 2
ERR_CLIENT  = 3
ERR_INFER   = 4

# ====== Trigger/Result files (devono combaciare con i client daemon) ======
LIP_TRIGGER_DEFAULT     = "/tmp/lip_go"
LIP_RESULT_DEFAULT      = "/tmp/lip_res"
BODY_TRIGGER_DEFAULT    = "/tmp/body_go"
BODY_RESULT_DEFAULT     = "/tmp/body_res"
BODYCL_TRIGGER_DEFAULT  = "/tmp/bodycl_go"
BODYCL_RESULT_DEFAULT   = "/tmp/bodycl_res"

# ====== pymodbus 2.5.3 ======
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock

# ---------- DataBlock con logging ----------
class LoggingDataBlock(ModbusSequentialDataBlock):
    """DataBlock che logga ogni write del client (PLC) sugli HR."""
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

# ---------- Helpers HR ----------
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
    print(">> Keyboard: '1' = START_CAPTURE, 'r' = ACK_RESET, 'q' = quit")
    while True:
        try:
            s = input().strip().lower()
        except EOFError:
            break
        if s in ("1","r","q"):
            q.put(s)
        else:
            print("  (unknown command, use 1/r/q)")

# ---------- File helpers ----------
def touch(path: str):
    with open(path, "w") as f:
        f.write("go\n")

def cleanup(paths: List[str]):
    for p in paths:
        try: os.remove(p)
        except: pass

# ---------- Attesa PARALLELA dei risultati ----------
def wait_results_parallel(pairs: List[Tuple[str, str]], timeout_s: float) -> Tuple[Dict[str, str], float]:
    """
    Attende in parallelo i risultati dei client daemon.
    pairs: lista di tuple (name, result_path), ad es. [("LIP", "/tmp/lip_res"), ("BODY", "/tmp/body_res"), ...]
    Ritorna: (tokens_dict, elapsed)
      tokens_dict[name] = "LIP_OK"/"LIP_NOK"/"BODY_OK"/"BODY_NOK"/"BODY_CL_OK"/"BODY_CL_NOK" oppure "<NAME>_ERROR"
    """
    t0 = time.time()
    seen: Dict[str, bool] = {name: False for name, _ in pairs}
    tokens: Dict[str, str] = {name: ""    for name, _ in pairs}

    path_by_name = {name: path for name, path in pairs}

    while (time.time() - t0) <= timeout_s:
        all_seen = True
        for name, path in pairs:
            if not seen[name]:
                all_seen = False
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            tok = (f.readline() or "").strip()
                        tokens[name] = tok
                        seen[name] = True
                        print(f"[ORCH] {name} result arrived: {tok}")
                    except:
                        pass
        if all_seen:
            break
        time.sleep(0.05)

    # fallback su timeout
    for name in seen:
        if not seen[name]:
            tokens[name] = f"{name}_ERROR"
            print(f"[ORCH] {name} result missing (timeout)")

    return tokens, (time.time() - t0)

# ---------- Core cycle ----------
def do_inference_cycle(ctx: ModbusServerContext, args):
    # Se già BUSY, ignora
    if get_hr(ctx, HR_BUSY, 1)[0] == 1:
        set_hr(ctx, HR_START, 0)  # consuma comunque
        print("[ORCH] START ignored: BUSY already 1")
        return

    print("[ORCH] START received -> triggering daemon clients")

    # Stato iniziale ciclo
    set_hr(ctx, HR_ERROR,    0)
    set_hr(ctx, HR_READY,    0)
    set_hr(ctx, HR_LIP,      0)
    set_hr(ctx, HR_BODY,     0)
    set_hr(ctx, HR_BODY_CL,  0)
    set_hr(ctx, HR_ERR_CODE, ERR_NONE)

    set_hr(ctx, HR_BUSY, 1)
    set_hr(ctx, HR_START, 0)  # consuma trigger
    print("[ORCH] HR: BUSY=1, READY=0, ERROR=0, RESULTS reset to 0")

    # pulisci vecchi risultati
    to_clean = [args.lip_result, args.body_result]
    pairs = [("LIP", args.lip_result), ("BODY", args.body_result)]
    if args.bodycl_result:  # opzionale terza camera
        to_clean.append(args.bodycl_result)
        pairs.append(("BODY_CL", args.bodycl_result))
    cleanup(to_clean)

    # trigger LIP/BODY/BODY_CL (se configurato)
    try:
        touch(args.lip_trigger);  print(f"[ORCH] -> LIP trigger: {args.lip_trigger}")
    except Exception as e:
        print(f"[ORCH] lip trigger error: {e}", file=sys.stderr)
    try:
        touch(args.body_trigger); print(f"[ORCH] -> BODY trigger: {args.body_trigger}")
    except Exception as e:
        print(f"[ORCH] body trigger error: {e}", file=sys.stderr)
    if args.bodycl_trigger:
        try:
            touch(args.bodycl_trigger); print(f"[ORCH] -> BODY_CL trigger: {args.bodycl_trigger}")
        except Exception as e:
            print(f"[ORCH] body_cl trigger error: {e}", file=sys.stderr)

    # attesa PARALLELA risultati
    tokens, elapsed = wait_results_parallel(pairs, args.timeout)
    lip_tok   = tokens.get("LIP", "LIP_ERROR")
    body_tok  = tokens.get("BODY", "BODY_ERROR")
    bodycl_tok= tokens.get("BODY_CL", "") if ("BODY_CL" in tokens) else ""
    print(f"[ORCH] tokens: lip={lip_tok}, body={body_tok}, body_cl={bodycl_tok or 'NA'}  elapsed={elapsed:.2f}s")

    # mapping token -> HR value
    def tok_to_hr(tok: str, prefix: str) -> int:
        if tok == f"{prefix}_OK":  return 1
        if tok == f"{prefix}_NOK": return 2
        return 0

    lip_hr   = tok_to_hr(lip_tok,   "LIP")
    body_hr  = tok_to_hr(body_tok,  "BODY")
    bodycl_hr= tok_to_hr(bodycl_tok,"BODY_CL") if bodycl_tok else 0

    set_hr(ctx, HR_LIP,     lip_hr)
    set_hr(ctx, HR_BODY,    body_hr)
    set_hr(ctx, HR_BODY_CL, bodycl_hr)

    # errore se QUALSIASI dei tre è 0 (mancante/invalid) tra quelli attesi
    err_code = ERR_NONE
    if (lip_hr == 0) or (body_hr == 0) or (("BODY_CL" in tokens) and bodycl_hr == 0):
        # se manca file entro timeout -> TIMEOUT, altrimenti INFER
        timeout_condition = False
        if lip_tok == "LIP_ERROR" and not os.path.exists(args.lip_result): timeout_condition = True
        if body_tok == "BODY_ERROR" and not os.path.exists(args.body_result): timeout_condition = True
        if ("BODY_CL" in tokens) and bodycl_tok == "BODY_CL_ERROR" and not os.path.exists(args.bodycl_result):
            timeout_condition = True
        err_code = ERR_TIMEOUT if timeout_condition else ERR_INFER

    set_hr(ctx, HR_ERR_CODE, err_code)

    # fine ciclo -> publish stato
    set_hr(ctx, HR_BUSY,  0)
    set_hr(ctx, HR_READY, 1)
    set_hr(ctx, HR_ERROR, 1 if err_code != ERR_NONE else 0)

    print(f"[ORCH] -> HRs: LIP={lip_hr}, BODY={body_hr}, BODY_CL={bodycl_hr}, ERR_CODE={err_code}, "
          f"BUSY=0 READY=1 ERROR={(1 if err_code else 0)}")
    print("[ORCH] Waiting for PLC ACK_RESET (HR[1] = 1)...")

    # attesa ACK o timeout
    t_ack = time.time()
    while True:
        if get_hr(ctx, HR_ACK, 1)[0] == 1:
            print("[ORCH] ACK received.")
            break
        if (time.time() - t_ack) > args.ack_timeout:
            print("[ORCH] ACK timeout -> auto-reset (safety).")
            break
        time.sleep(0.05)

    # reset post-ack
    set_hr(ctx, HR_ACK,      0)
    set_hr(ctx, HR_READY,    0)
    set_hr(ctx, HR_LIP,      0)
    set_hr(ctx, HR_BODY,     0)
    set_hr(ctx, HR_BODY_CL,  0)
    set_hr(ctx, HR_ERR_CODE, ERR_NONE)
    set_hr(ctx, HR_ERROR,    0)
    cleanup(to_clean)
    print("[ORCH] Reset done. Back to IDLE.")

def main():
    ap = argparse.ArgumentParser(description="Orchestratore Modbus (daemon) - tutto su HR")
    ap.add_argument("--bind-ip", default=BIND_IP)
    ap.add_argument("--bind-port", type=int, default=BIND_PORT)
    ap.add_argument("--timeout", type=float, default=RESULT_TIMEOUT_S)
    ap.add_argument("--ack-timeout", type=float, default=ACK_TIMEOUT_S)

    # paths trigger/result (devono combaciare con i 3 client daemon)
    ap.add_argument("--lip-trigger",   default=LIP_TRIGGER_DEFAULT)
    ap.add_argument("--lip-result",    default=LIP_RESULT_DEFAULT)
    ap.add_argument("--body-trigger",  default=BODY_TRIGGER_DEFAULT)
    ap.add_argument("--body-result",   default=BODY_RESULT_DEFAULT)
    ap.add_argument("--bodycl-trigger", default=BODYCL_TRIGGER_DEFAULT)
    ap.add_argument("--bodycl-result",  default=BODYCL_RESULT_DEFAULT)
    args = ap.parse_args()

    # Datastore: usiamo solo HR (ma creiamo anche gli altri blocchi per completezza)
    hr_block   = LoggingDataBlock(0, [0]*32, "HR")
    di_block   = LoggingDataBlock(0, [0]*1,  "DI")    # inutilizzato
    co_block   = LoggingDataBlock(0, [0]*1,  "COIL")  # inutilizzato
    ir_block   = LoggingDataBlock(0, [0]*1,  "IR")    # inutilizzato

    def on_write(name, address, values):
        if name == "HR":
            print(f"[PLC->MODBUS] WRITE {name} addr={address} values={values}")

    hr_block.on_write = on_write

    store = ModbusSlaveContext(di=di_block, co=co_block, hr=hr_block, ir=ir_block)
    context = ModbusServerContext(slaves=store, single=True)

    # Stato iniziale
    set_hr(context, HR_START,    0)
    set_hr(context, HR_ACK,      0)
    set_hr(context, HR_BUSY,     0)
    set_hr(context, HR_READY,    0)
    set_hr(context, HR_ERROR,    0)
    set_hr(context, HR_LIP,      0)
    set_hr(context, HR_BODY,     0)
    set_hr(context, HR_BODY_CL,  0)
    set_hr(context, HR_ERR_CODE, ERR_NONE)

    # Avvia server
    start_modbus_server(context, args.bind_ip, args.bind_port)
    print(f"[SERVER] Modbus TCP listening on {args.bind_ip}:{args.bind_port}")
    print("         HR map: 40001 START, 40002 ACK, 40003 BUSY, 40004 READY, 40005 ERROR, "
          "40006 LIP, 40007 BODY, 40008 BODY_CL, 40009 ERR_CODE")
    print("         Daemon mode: make sure LIP/BODY(/BODY_CL) clients are running and paths match.")
    print("[ORCH] Waiting for START (HR[0]=1) or keyboard '1'...")

    # Tastiera per simulazione senza PLC
    q = Queue()
    tk = threading.Thread(target=keyboard_thread, args=(q,), daemon=True)
    tk.start()

    try:
        while True:
            # tastiera
            try:
                cmd = q.get_nowait()
                if cmd == "q":
                    print("[ORCH] Quit requested.")
                    break
                elif cmd == "1":
                    print("[SIM] Keyboard -> START_CAPTURE")
                    set_hr(context, HR_START, 1)
                elif cmd == "r":
                    print("[SIM] Keyboard -> ACK_RESET")
                    set_hr(context, HR_ACK, 1)
            except Empty:
                pass

            # trigger da PLC
            if get_hr(context, HR_START, 1)[0] == 1:
                do_inference_cycle(context, args)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[ORCH] CTRL-C -> exit.")
    finally:
        print("[ORCH] Bye.")

if __name__ == "__main__":
    main()

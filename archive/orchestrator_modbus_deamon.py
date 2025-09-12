#!/usr/bin/env python3
"""
Orchestrator MODBUS TCP (daemon clients)
- Jetson = Modbus TCP Server (bind 0.0.0.0:5020 by default)
- PLC (192.168.1.5) = Modbus TCP Client

Mapping (human addressing; internal 0-based indices in parentheses)
  Coils (PLC -> Jetson):
    00001 (0) START_CAPTURE  : 0/1
    00002 (1) ACK_RESET      : 0/1
  Discrete Inputs (Jetson -> PLC):
    10001 (0) BUSY           : 0/1
    10002 (1) RESULT_READY   : 0/1
    10003 (2) ERROR          : 0/1
  Holding Registers (Jetson -> PLC):
    40001 (0) LIP_RESULT     : 0=UNK, 1=OK, 2=NOK
    40002 (1) BODY_RESULT    : 0=UNK, 1=OK, 2=NOK
    40003 (2) ERROR_CODE     : 0=NONE, 2=TIMEOUT, 3=CLIENT_FAIL, 4=INFERENCE_FAIL

Flow (daemon):
- lip_client.py and body_client.py are already running with --daemon and windows open
- When PLC writes START_CAPTURE=1:
    * Jetson consumes START (sets it back to 0)
    * Sets BUSY=1, clears previous results
    * Creates trigger files for both clients
    * Waits results (or timeout)
    * Publishes HR/DI
    * Waits PLC ACK_RESET=1 (or timeout), then resets and returns to IDLE
"""

import os, sys, time, threading, argparse
from typing import List

# ====== TIMEOUTS ======
RESULT_TIMEOUT_S = 15.0   # wait for daemon results (6s window + margin)
ACK_TIMEOUT_S    = 30.0   # wait for PLC ACK before auto-reset

# ====== MODBUS SERVER ======
BIND_IP   = "0.0.0.0"     # Jetson bind IP
BIND_PORT = 5020          # Modbus TCP port

# ====== ADDRESS MAP (0-based) ======
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

# ====== Trigger/Result files (must match daemon clients) ======
LIP_TRIGGER_DEFAULT  = "/tmp/lip_go"
LIP_RESULT_DEFAULT   = "/tmp/lip_res"
BODY_TRIGGER_DEFAULT = "/tmp/body_go"
BODY_RESULT_DEFAULT  = "/tmp/body_res"

# ====== pymodbus 2.5.3 ======
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock

# ---------- Data Blocks ----------
class SilentDataBlock(ModbusSequentialDataBlock):
    """Plain DataBlock; logging is done in the orchestrator prints."""
    pass

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

def start_modbus_server(context: ModbusServerContext, bind_ip: str, bind_port: int):
    t = threading.Thread(
        target=StartTcpServer,
        kwargs={"context": context, "address": (bind_ip, bind_port)},
        daemon=True
    )
    t.start()
    return t

# ---------- File helpers ----------
def touch(path: str):
    with open(path, "w") as f:
        f.write("go\n")

def wait_result(path: str, timeout_s: float) -> str:
    """Wait for result file and return first line stripped; '' on timeout."""
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

def cleanup(paths: List[str]):
    for p in paths:
        try: os.remove(p)
        except: pass

# ---------- Inference cycle ----------
def do_inference_cycle(ctx: ModbusServerContext, args):
    # Reject if already BUSY
    if get_di(ctx, DI_BUSY, 1)[0] == 1:
        set_coil(ctx, COIL_START, 0)
        print("[ORCH] START ignored: already BUSY")
        return

    print("[ORCH] START received -> trigger daemon clients")

    # Prepare state
    set_di(ctx, DI_ERROR, 0)
    set_di(ctx, DI_READY, 0)
    set_hr(ctx, HR_LIP,  0)
    set_hr(ctx, HR_BODY, 0)
    set_hr(ctx, HR_ERR,  ERR_NONE)

    set_di(ctx, DI_BUSY, 1)
    set_coil(ctx, COIL_START, 0)  # consume trigger
    print("[ORCH] BUSY=1, READY=0, HR(LIP,BODY,ERR)=(0,0,0)")

    # Clear old results
    cleanup([args.lip_result, args.body_result])

    # Trigger both daemon clients
    try:
        touch(args.lip_trigger)
        print(f"[ORCH] -> LIP trigger: {args.lip_trigger}")
    except Exception as e:
        print(f"[ORCH] lip trigger error: {e}", file=sys.stderr)
    try:
        touch(args.body_trigger)
        print(f"[ORCH] -> BODY trigger: {args.body_trigger}")
    except Exception as e:
        print(f"[ORCH] body trigger error: {e}", file=sys.stderr)

    # Wait results
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

    # Error code
    err_code = ERR_NONE
    if lip_hr == 0 or body_hr == 0:
        # If no token within timeout -> TIMEOUT, else generic INFER error
        if (lip_tok == "LIP_ERROR" and not os.path.exists(args.lip_result)) or \
           (body_tok == "BODY_ERROR" and not os.path.exists(args.body_result)):
            err_code = ERR_TIMEOUT
        else:
            err_code = ERR_INFER
    set_hr(ctx, HR_ERR, err_code)

    # Done: publish DI
    set_di(ctx, DI_BUSY, 0)
    set_di(ctx, DI_READY, 1)
    set_di(ctx, DI_ERROR, 1 if err_code != ERR_NONE else 0)

    print(f"[ORCH] -> HR(LIP,BODY,ERR)=({lip_hr},{body_hr},{err_code}), DI: BUSY=0 READY=1 ERROR={(1 if err_code else 0)}")
    print("[ORCH] Waiting for PLC ACK (Coil 00002 = 1)...")

    # Wait for ACK_RESET or timeout
    t_ack = time.time()
    while True:
        if get_coil(ctx, COIL_ACK, 1)[0] == 1:
            print("[ORCH] ACK received.")
            break
        if (time.time() - t_ack) > args.ack_timeout:
            print("[ORCH] ACK timeout -> auto-reset (safety).")
            break
        time.sleep(0.05)

    # Clear & reset for next cycle
    set_coil(ctx, COIL_ACK, 0)
    set_di(ctx, DI_READY, 0)
    set_hr(ctx, HR_LIP,  0)
    set_hr(ctx, HR_BODY, 0)
    set_hr(ctx, HR_ERR,  ERR_NONE)
    set_di(ctx, DI_ERROR, 0)
    cleanup([args.lip_result, args.body_result])
    print("[ORCH] Reset done. Back to IDLE.")

def main():
    ap = argparse.ArgumentParser(description="Orchestrator Modbus TCP (daemon clients)")
    ap.add_argument("--bind-ip", default=BIND_IP, help="Bind IP (Jetson)")
    ap.add_argument("--bind-port", type=int, default=BIND_PORT, help="Modbus TCP port")
    ap.add_argument("--timeout", type=float, default=RESULT_TIMEOUT_S, help="Result wait timeout (s)")
    ap.add_argument("--ack-timeout", type=float, default=ACK_TIMEOUT_S, help="ACK wait timeout (s)")

    # Trigger/Result paths (must match clients)
    ap.add_argument("--lip-trigger",  default=LIP_TRIGGER_DEFAULT)
    ap.add_argument("--lip-result",   default=LIP_RESULT_DEFAULT)
    ap.add_argument("--body-trigger", default=BODY_TRIGGER_DEFAULT)
    ap.add_argument("--body-result",  default=BODY_RESULT_DEFAULT)
    args = ap.parse_args()

    # Datastore
    di_block   = SilentDataBlock(0, [0]*8)
    coil_block = SilentDataBlock(0, [0]*8)
    hr_block   = SilentDataBlock(0, [0]*16)
    ir_block   = SilentDataBlock(0, [0]*16)

    store = ModbusSlaveContext(di=di_block, co=coil_block, hr=hr_block, ir=ir_block)
    context = ModbusServerContext(slaves=store, single=True)

    # Initialize state
    set_di(context, DI_BUSY, 0)
    set_di(context, DI_READY, 0)
    set_di(context, DI_ERROR, 0)
    set_hr(context, HR_LIP,  0)
    set_hr(context, HR_BODY, 0)
    set_hr(context, HR_ERR,  ERR_NONE)
    set_coil(context, COIL_START, 0)
    set_coil(context, COIL_ACK,   0)

    # Start server in background
    start_modbus_server(context, args.bind_ip, args.bind_port)
    print(f"[SERVER] Modbus TCP listening on {args.bind_ip}:{args.bind_port}")
    print("         Coils: 00001 START, 00002 ACK | DI: 10001 BUSY, 10002 READY, 10003 ERROR | HR: 40001 LIP, 40002 BODY, 40003 ERR")
    print("         Daemon mode: make sure lip/body clients are running with matching trigger/result paths.")

    try:
        while True:
            if get_coil(context, COIL_START, 1)[0] == 1:
                do_inference_cycle(context, args)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[ORCH] Stopping...")
    finally:
        print("[ORCH] Bye.")

if __name__ == "__main__":
    main()

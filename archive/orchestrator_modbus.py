#!/usr/bin/env python3
"""
Orchestrator MODBUS TCP (production)
- Jetson = Modbus TCP Server (bind 0.0.0.0:5020 by default)
- PLC (192.168.1.5) = Modbus TCP Client

Mapping (human addressing; internal indices in parentheses):
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
"""

import sys
import time
import threading
import argparse
import subprocess
from typing import Optional, Tuple, List

# ====== PATHS (adjust to your environment) ======
PY_INFER   = "/home/bewater/big400_bottle_inspection/inference_clients/.venv/bin/python"
LIP_SCRIPT = "/home/bewater/big400_bottle_inspection/inference_clients/lip_client.py"
BODY_SCRIPT= "/home/bewater/big400_bottle_inspection/inference_clients/body_client.py"

LIP_EXTRA_ARGS  = ["--cam-index","0"]   # adjust if needed
BODY_EXTRA_ARGS = ["--cam-index","1"]

# ====== TIMEOUTS ======
CLIENT_TIMEOUT_S = 15.0   # each vision client (6s window + margin)
ACK_TIMEOUT_S    = 30.0   # how long to wait for PLC ACK before auto-reset (s)

# ====== MODBUS SERVER ======
BIND_IP   = "0.0.0.0"   # Jetson bind IP
BIND_PORT = 5020        # Modbus TCP port

# ====== ADDRESS MAP (0-based index) ======
COIL_START = 0
COIL_ACK   = 1

DI_BUSY    = 0
DI_READY   = 1
DI_ERROR   = 2

HR_LIP     = 0
HR_BODY    = 1
HR_ERR     = 2

ERR_NONE   = 0
ERR_TIMEOUT= 2
ERR_CLIENT = 3
ERR_INFER  = 4

# ====== pymodbus 2.5.3 ======
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock

# ---------- Data Blocks ----------
class SilentDataBlock(ModbusSequentialDataBlock):
    """Plain DataBlock; we keep logging in the orchestrator, not on every write."""
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

# ---------- Vision client runner ----------
def run_client(cmd: list, expected_prefix: str, timeout_s: float):
    """
    Run a vision client process.
    Returns (token, stderr, returncode); token in {LIP_OK, LIP_NOK, BODY_OK, BODY_NOK} or None.
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
            err = f"[{expected_prefix}] rc={rc} unexpected output: '{out}'"
        return None, err, rc
    else:
        return None, f"[{expected_prefix}] unexpected output rc=0: '{out}'", rc

# ---------- Cycle ----------
def do_inference_cycle(ctx: ModbusServerContext, args):
    # Reject if already BUSY
    if get_di(ctx, DI_BUSY, 1)[0] == 1:
        # consume START just in case and ignore
        set_coil(ctx, COIL_START, 0)
        print("[ORCH] START ignored: already BUSY")
        return

    print("[ORCH] START received -> running inference cycle")
    # Prepare state
    set_di(ctx, DI_ERROR, 0)
    set_di(ctx, DI_READY, 0)
    set_hr(ctx, HR_LIP,  0)
    set_hr(ctx, HR_BODY, 0)
    set_hr(ctx, HR_ERR,  ERR_NONE)

    set_di(ctx, DI_BUSY, 1)
    set_coil(ctx, COIL_START, 0)  # consume trigger

    # Build commands
    lip_cmd  = [args.py_infer, args.lip_script]  + args.lip_args
    body_cmd = [args.py_infer, args.body_script] + args.body_args

    # Run in parallel
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

    lip_out  = results["LIP"]  or "LIP_ERROR"
    body_out = results["BODY"] or "BODY_ERROR"
    if errors["LIP"]:
        print(f"[LIP][stderr] {errors['LIP']}", file=sys.stderr)
    if errors["BODY"]:
        print(f"[BODY][stderr] {errors['BODY']}", file=sys.stderr)
    print(f"[ORCH] results: {lip_out}, {body_out}  elapsed={elapsed:.2f}s")

    # HR mapping
    def tok_to_hr(tok, prefix):
        if tok == f"{prefix}_OK":  return 1
        if tok == f"{prefix}_NOK": return 2
        return 0

    lip_hr  = tok_to_hr(lip_out,  "LIP")
    body_hr = tok_to_hr(body_out, "BODY")
    set_hr(ctx, HR_LIP,  lip_hr)
    set_hr(ctx, HR_BODY, body_hr)

    # Error code
    err_code = ERR_NONE
    if lip_hr == 0 or body_hr == 0:
        if rcodes["LIP"] == 124 or rcodes["BODY"] == 124:
            err_code = ERR_TIMEOUT
        elif rcodes["LIP"] != 0 or rcodes["BODY"] != 0:
            err_code = ERR_CLIENT
        else:
            err_code = ERR_INFER
    set_hr(ctx, HR_ERR, err_code)

    # Done -> update DI
    set_di(ctx, DI_BUSY, 0)
    set_di(ctx, DI_READY, 1)
    set_di(ctx, DI_ERROR, 1 if err_code != ERR_NONE else 0)

    print(f"[ORCH] HR(LIP,BODY,ERR)=({lip_hr},{body_hr},{err_code})  DI: BUSY=0 READY=1 ERROR={1 if err_code else 0}")
    print("[ORCH] Waiting for PLC ACK (Coil 00002=1)...")

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

    # Clear & reset
    set_coil(ctx, COIL_ACK, 0)
    set_di(ctx, DI_READY, 0)
    set_hr(ctx, HR_LIP,  0)
    set_hr(ctx, HR_BODY, 0)
    set_hr(ctx, HR_ERR,  ERR_NONE)
    set_di(ctx, DI_ERROR, 0)
    print("[ORCH] Reset done. Back to IDLE.")

def main():
    ap = argparse.ArgumentParser(description="Orchestrator Modbus TCP (production)")
    ap.add_argument("--bind-ip", default=BIND_IP, help="Bind IP (Jetson).")
    ap.add_argument("--bind-port", type=int, default=BIND_PORT, help="Modbus TCP port.")
    ap.add_argument("--py-infer", default=PY_INFER, help="Python path for inference venv.")
    ap.add_argument("--lip-script", default=LIP_SCRIPT, help="Path to lip_client.py.")
    ap.add_argument("--body-script", default=BODY_SCRIPT, help="Path to body_client.py.")
    ap.add_argument("--timeout", type=float, default=CLIENT_TIMEOUT_S, help="Client timeout (s).")
    ap.add_argument("--ack-timeout", type=float, default=ACK_TIMEOUT_S, help="ACK wait timeout (s).")
    ap.add_argument("--lip-args",  default=",".join(LIP_EXTRA_ARGS),  help="CSV extra args for lip client.")
    ap.add_argument("--body-args", default=",".join(BODY_EXTRA_ARGS), help="CSV extra args for body client.")
    args = ap.parse_args()

    args.lip_args  = [s for s in (args.lip_args.split(",") if args.lip_args else []) if s]
    args.body_args = [s for s in (args.body_args.split(",") if args.body_args else []) if s]

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

    try:
        while True:
            # Poll for START trigger
            if get_coil(context, COIL_START, 1)[0] == 1:
                do_inference_cycle(context, args)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[ORCH] Stopping...")
    finally:
        print("[ORCH] Bye.")

if __name__ == "__main__":
    main()


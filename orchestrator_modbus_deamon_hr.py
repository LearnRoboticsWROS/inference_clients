#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator Modbus TCP (DAEMON) - HR-only mapping (production)
Jetson = Modbus TCP Server (default 0.0.0.0:5020)
PLC    = Modbus TCP Client

HR map (human addressing -> internal 0-based index):
  40001 (0) START_CAPTURE   : 0/1   (PLC -> Jetson)
  40002 (1) ACK_RESET       : 0/1   (PLC -> Jetson)
  40003 (2) BUSY            : 0/1   (Jetson -> PLC)
  40004 (3) RESULT_READY    : 0/1   (Jetson -> PLC)
  40005 (4) ERROR           : 0/1   (Jetson -> PLC)
  40006 (5) LIP_RESULT      : 0=UNK, 1=OK, 2=NOK
  40007 (6) BODY_RESULT     : 0=UNK, 1=OK, 2=NOK
  40008 (7) BODY_CL_RESULT  : 0=UNK, 1=OK, 2=NOK   (optional 3rd camera)
  40009 (8) ERROR_CODE      : 0=NONE, 2=TIMEOUT, 3=CLIENT_FAIL, 4=INFERENCE_FAIL

Flow (daemon):
- Vision clients (lip/body[/body_cl]) are already running with --daemon and open windows
- PLC writes START_CAPTURE=1 -> cycle starts
- Orchestrator:
    * consumes START (back to 0)
    * BUSY=1, READY=0, ERROR=0, results reset
    * creates trigger files for LIP/BODY(/BODY_CL)
    * waits all result files in parallel (within timeout)
    * publishes results to HRs, sets READY=1, BUSY=0, sets ERROR/ERR_CODE if needed
    * waits PLC ACK_RESET=1 (or timeout), then resets and returns to IDLE
"""

import os
import sys
import time
import argparse
import threading
from typing import List, Dict, Tuple

# ====== DEFAULT TIMEOUTS ======
RESULT_TIMEOUT_S = 8.0   # suggest 8s with 6s inference window + buffer
ACK_TIMEOUT_S    = 4.0   # suggest 4s within your 3s PLC window + small buffer

# ====== SERVER BIND ======
BIND_IP   = "0.0.0.0"
BIND_PORT = 5020

# ====== HR INDICES (0-based) ======
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

# ====== Trigger/Result files (must match daemon clients) ======
LIP_TRIGGER_DEFAULT     = "/tmp/lip_go"
LIP_RESULT_DEFAULT      = "/tmp/lip_res"
BODY_TRIGGER_DEFAULT    = "/tmp/body_go"
BODY_RESULT_DEFAULT     = "/tmp/body_res"
BODYCL_TRIGGER_DEFAULT  = "/tmp/bodycl_go"
BODYCL_RESULT_DEFAULT   = "/tmp/bodycl_res"

# ====== pymodbus 2.5.3 ======
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock

# ---------- Minimal DataBlock ----------
class SilentDataBlock(ModbusSequentialDataBlock):
    pass

# ---------- HR helpers ----------
def set_hr(ctx: ModbusServerContext, idx: int, val: int):
    ctx[0x00].setValues(3, idx, [int(val)])

def get_hr(ctx: ModbusServerContext, idx: int, count: int = 1) -> List[int]:
    return ctx[0x00].getValues(3, idx, count)

# ---------- Start server in background ----------
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

def cleanup(paths: List[str]):
    for p in paths:
        try: os.remove(p)
        except: pass

# ---------- Parallel wait for results ----------
def wait_results_parallel(pairs: List[Tuple[str, str]], timeout_s: float) -> Tuple[Dict[str, str], float]:
    """
    Waits for all result files in parallel.
    pairs: list of (name, result_path) -> e.g. [("LIP","/tmp/lip_res"), ("BODY","/tmp/body_res"), ...]
    Returns: (tokens_dict, elapsed_seconds)
      tokens_dict[name] in {"<NAME>_OK","<NAME>_NOK"} or "<NAME>_ERROR" on timeout
    """
    t0 = time.time()
    seen: Dict[str, bool] = {name: False for name, _ in pairs}
    tokens: Dict[str, str] = {name: ""    for name, _ in pairs}

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
                    except:
                        pass
        if all_seen:
            break
        time.sleep(0.05)

    # fallback on timeout
    for name in seen:
        if not seen[name]:
            tokens[name] = f"{name}_ERROR"

    return tokens, (time.time() - t0)

# ---------- One inference cycle ----------
def do_inference_cycle(ctx: ModbusServerContext, args):
    # Ignore if BUSY
    if get_hr(ctx, HR_BUSY, 1)[0] == 1:
        set_hr(ctx, HR_START, 0)  # consume anyway
        return

    # Prepare state
    set_hr(ctx, HR_ERROR,    0)
    set_hr(ctx, HR_READY,    0)
    set_hr(ctx, HR_LIP,      0)
    set_hr(ctx, HR_BODY,     0)
    set_hr(ctx, HR_BODY_CL,  0)
    set_hr(ctx, HR_ERR_CODE, ERR_NONE)

    set_hr(ctx, HR_BUSY, 1)
    set_hr(ctx, HR_START, 0)  # consume trigger

    # Clear previous results
    to_clean = [args.lip_result, args.body_result]
    pairs = [("LIP", args.lip_result), ("BODY", args.body_result)]
    if args.bodycl_result:
        to_clean.append(args.bodycl_result)
        pairs.append(("BODY_CL", args.bodycl_result))
    cleanup(to_clean)

    # Trigger DAEMON clients
    try: touch(args.lip_trigger)
    except Exception as e: print(f"[lip trigger error] {e}", file=sys.stderr)
    try: touch(args.body_trigger)
    except Exception as e: print(f"[body trigger error] {e}", file=sys.stderr)
    if args.bodycl_trigger:
        try: touch(args.bodycl_trigger)
        except Exception as e: print(f"[body_cl trigger error] {e}", file=sys.stderr)

    # Wait results (parallel)
    tokens, _elapsed = wait_results_parallel(pairs, args.timeout)
    lip_tok   = tokens.get("LIP", "LIP_ERROR")
    body_tok  = tokens.get("BODY", "BODY_ERROR")
    bodycl_tok= tokens.get("BODY_CL", "") if ("BODY_CL" in tokens) else ""
    print(f"[ORCH] inference elapsed = {_elapsed:.3f}s", flush=True)


    # token -> HR mapping
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

    # Error code if any expected result is missing/invalid
    err_code = ERR_NONE
    if (lip_hr == 0) or (body_hr == 0) or (("BODY_CL" in tokens) and bodycl_hr == 0):
        # If missing file within timeout -> TIMEOUT, else generic INFER
        timeout_condition = False
        if lip_tok == "LIP_ERROR" and not os.path.exists(args.lip_result): timeout_condition = True
        if body_tok == "BODY_ERROR" and not os.path.exists(args.body_result): timeout_condition = True
        if ("BODY_CL" in tokens) and bodycl_tok == "BODY_CL_ERROR" and not os.path.exists(args.bodycl_result):
            timeout_condition = True
        err_code = ERR_TIMEOUT if timeout_condition else ERR_INFER

    set_hr(ctx, HR_ERR_CODE, err_code)

    # Publish end-of-cycle state
    set_hr(ctx, HR_BUSY,  0)
    set_hr(ctx, HR_READY, 1)
    set_hr(ctx, HR_ERROR, 1 if err_code != ERR_NONE else 0)

    # Wait PLC ACK (or timeout), then reset
    t_ack = time.time()
    while True:
        if get_hr(ctx, HR_ACK, 1)[0] == 1:
            break
        if (time.time() - t_ack) > args.ack_timeout:
            break
        time.sleep(0.02)

    # Reset for next cycle
    set_hr(ctx, HR_ACK,      0)
    set_hr(ctx, HR_READY,    0)
    set_hr(ctx, HR_LIP,      0)
    set_hr(ctx, HR_BODY,     0)
    set_hr(ctx, HR_BODY_CL,  0)
    set_hr(ctx, HR_ERR_CODE, ERR_NONE)
    set_hr(ctx, HR_ERROR,    0)
    cleanup(to_clean)

def main():
    ap = argparse.ArgumentParser(description="Orchestrator Modbus (daemon) - HR only (production)")
    ap.add_argument("--bind-ip", default=BIND_IP)
    ap.add_argument("--bind-port", type=int, default=BIND_PORT)
    ap.add_argument("--timeout", type=float, default=RESULT_TIMEOUT_S)
    ap.add_argument("--ack-timeout", type=float, default=ACK_TIMEOUT_S)
    # paths trigger/result (must match daemon clients)
    ap.add_argument("--lip-trigger",   default=LIP_TRIGGER_DEFAULT)
    ap.add_argument("--lip-result",    default=LIP_RESULT_DEFAULT)
    ap.add_argument("--body-trigger",  default=BODY_TRIGGER_DEFAULT)
    ap.add_argument("--body-result",   default=BODY_RESULT_DEFAULT)
    ap.add_argument("--bodycl-trigger", default=BODYCL_TRIGGER_DEFAULT)
    ap.add_argument("--bodycl-result",  default=BODYCL_RESULT_DEFAULT)
    args = ap.parse_args()

    # Minimal datastore (HR only actually used)
    hr_block = SilentDataBlock(0, [0]*32)
    di_block = SilentDataBlock(0, [0]*1)   # unused
    co_block = SilentDataBlock(0, [0]*1)   # unused
    ir_block = SilentDataBlock(0, [0]*1)   # unused

    store = ModbusSlaveContext(di=di_block, co=co_block, hr=hr_block, ir=ir_block)
    context = ModbusServerContext(slaves=store, single=True)

    # Initial state
    set_hr(context, HR_START,    0)
    set_hr(context, HR_ACK,      0)
    set_hr(context, HR_BUSY,     0)
    set_hr(context, HR_READY,    0)
    set_hr(context, HR_ERROR,    0)
    set_hr(context, HR_LIP,      0)
    set_hr(context, HR_BODY,     0)
    set_hr(context, HR_BODY_CL,  0)
    set_hr(context, HR_ERR_CODE, ERR_NONE)

    # Start Modbus server
    start_modbus_server(context, args.bind_ip, args.bind_port)
    print(f"[SERVER] Modbus TCP listening on {args.bind_ip}:{args.bind_port}")
    print("         HR map: 40001 START, 40002 ACK, 40003 BUSY, 40004 READY, 40005 ERROR, "
          "40006 LIP, 40007 BODY, 40008 BODY_CL, 40009 ERR_CODE")
    print("         Daemon mode: ensure LIP/BODY(/BODY_CL) clients are running and paths match.")

    try:
        while True:
            if get_hr(context, HR_START, 1)[0] == 1:
                do_inference_cycle(context, args)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[ORCH] Stopping...")
    finally:
        print("[ORCH] Bye.")

if __name__ == "__main__":
    main()


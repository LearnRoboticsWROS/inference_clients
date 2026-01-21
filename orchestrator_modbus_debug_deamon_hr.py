#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Modbus Orchestrator (daemon) - Holding Registers (HR)
Jetson = Modbus TCP Server (0.0.0.0:5020)
PLC    = Modbus TCP Client
"""

import os, sys, time, threading, argparse
from queue import Queue, Empty
from typing import List, Dict, Tuple
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock

# ====== TIMEOUTS ======
RESULT_TIMEOUT_S = 2.0
ACK_TIMEOUT_S    = 2.0

# ====== MODBUS ======
BIND_IP   = "0.0.0.0"
BIND_PORT = 5020

# ====== HR INDICES (0-based) ======
HR_START       = 0
HR_ACK         = 1
HR_BUSY        = 2
HR_READY       = 3
HR_ERROR       = 4
HR_LIP         = 5
HR_BODY        = 6
HR_BODY_CL     = 7
HR_ERR_CODE    = 8

# ====== ERROR CODES ======
ERR_NONE    = 0
ERR_TIMEOUT = 2
ERR_CLIENT  = 3
ERR_INFER   = 4

# ====== Trigger/Result files ======
LIP_TRIGGER_DEFAULT     = "/tmp/lip_go"
LIP_RESULT_DEFAULT      = "/tmp/lip_res"
BODY_TRIGGER_DEFAULT    = "/tmp/body_go"
BODY_RESULT_DEFAULT     = "/tmp/body_res"
BODYCL_TRIGGER_DEFAULT  = "/tmp/bodycl_go"
BODYCL_RESULT_DEFAULT   = "/tmp/bodycl_res"

# ---------- DataBlock with logging ----------
class LoggingDataBlock(ModbusSequentialDataBlock):
    def __init__(self, address: int, values: List[int], name: str):
        super().__init__(address, values)
        self.name = name
        self.on_write = None

    def setValues(self, address, values):
        super().setValues(address, values)
        if self.on_write:
            try:
                self.on_write(self.name, address, list(values))
            except Exception as e:
                print(f"[DataBlock][{self.name}] on_write error: {e}", file=sys.stderr)

# ---------- Helpers ----------
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

# ---------- Keyboard ----------
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

# ---------- Wait results asynchronously with fast HR updates ----------
def wait_results_async(ctx, pairs: List[Tuple[str,str]], timeout_s: float) -> Dict[str,str]:
    tokens: Dict[str,str] = {name: "" for name,_ in pairs}
    seen: Dict[str,bool] = {name: False for name,_ in pairs}
    t0 = time.time()
    while (time.time() - t0) <= timeout_s:
        all_seen = True
        for name, path in pairs:
            if not seen[name]:
                all_seen = False
                set_hr(ctx, HR_BUSY, 1)  # indicate busy while waiting
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            tok = (f.readline() or "").strip()
                        tokens[name] = tok
                        seen[name] = True
                        # fast HR update per client
                        if name=="LIP":
                            set_hr(ctx, HR_LIP, 1 if tok=="LIP_OK" else 2 if tok=="LIP_NOK" else 0)
                        elif name=="BODY":
                            set_hr(ctx, HR_BODY, 1 if tok=="BODY_OK" else 2 if tok=="BODY_NOK" else 0)
                        elif name=="BODY_CL":
                            set_hr(ctx, HR_BODY_CL, 1 if tok=="BODY_CL_OK" else 2 if tok=="BODY_CL_NOK" else 0)
                        # indicate READY immediately
                        set_hr(ctx, HR_READY, 1)
                        print(f"[ORCH] {name} result: {tok}")
                    except: pass
        if all_seen: break
        time.sleep(0.00001)  # very short sleep for fast updates

    # fallback for missing results
    for name in seen:
        if not seen[name]:
            tokens[name] = f"{name}_ERROR"
            print(f"[ORCH] {name} result missing (timeout)")
    return tokens

# ---------- Core cycle ----------
def do_inference_cycle(ctx: ModbusServerContext, args):
    if get_hr(ctx, HR_BUSY,1)[0]==1:
        set_hr(ctx, HR_START,0)
        print("[ORCH] START ignored: BUSY already 1")
        return

    print("[ORCH] START received -> triggering clients")
    # reset HRs
    for idx in range(HR_ERROR, HR_ERR_CODE+1):
        set_hr(ctx, idx, 0)
    set_hr(ctx, HR_BUSY,1)
    set_hr(ctx, HR_START,0)
    set_hr(ctx, HR_READY,0)

    # cleanup previous results
    to_clean = [args.lip_result, args.body_result]
    pairs = [("LIP", args.lip_result), ("BODY", args.body_result)]
    if args.bodycl_result:
        to_clean.append(args.bodycl_result)
        pairs.append(("BODY_CL", args.bodycl_result))
    cleanup(to_clean)

    # trigger clients
    try: touch(args.lip_trigger); print(f"[ORCH] -> LIP trigger")
    except: pass
    try: touch(args.body_trigger); print(f"[ORCH] -> BODY trigger")
    except: pass
    if args.bodycl_trigger:
        try: touch(args.bodycl_trigger); print(f"[ORCH] -> BODY_CL trigger")
        except: pass

    # async wait with fast HR updates
    tokens = wait_results_async(ctx, pairs, args.timeout)

    # determine error code
    lip_hr = get_hr(ctx, HR_LIP)[0]
    body_hr = get_hr(ctx, HR_BODY)[0]
    bodycl_hr = get_hr(ctx, HR_BODY_CL)[0]
    err_code = ERR_NONE
    if lip_hr==0 or body_hr==0 or (("BODY_CL" in tokens) and bodycl_hr==0):
        timeout_condition = False
        if tokens.get("LIP")=="LIP_ERROR" and not os.path.exists(args.lip_result): timeout_condition=True
        if tokens.get("BODY")=="BODY_ERROR" and not os.path.exists(args.body_result): timeout_condition=True
        if ("BODY_CL" in tokens) and tokens.get("BODY_CL")=="BODY_CL_ERROR" and not os.path.exists(args.bodycl_result):
            timeout_condition=True
        err_code = ERR_TIMEOUT if timeout_condition else ERR_INFER
    set_hr(ctx, HR_ERR_CODE, err_code)

    # finalize
    set_hr(ctx, HR_BUSY,0)
    set_hr(ctx, HR_ERROR,1 if err_code!=ERR_NONE else 0)
    print(f"[ORCH] -> HRs: LIP={lip_hr}, BODY={body_hr}, BODY_CL={bodycl_hr}, ERR_CODE={err_code}")

    # wait for ACK in background thread
    def ack_wait_thread():
        t_ack = time.time()
        while True:
            if get_hr(ctx, HR_ACK,1)[0]==1:
                print("[ORCH] ACK received")
                break
            if (time.time()-t_ack) > args.ack_timeout:
                print("[ORCH] ACK timeout -> auto-reset")
                break
            time.sleep(0.00001)  # very fast check

        # reset HRs
        for idx in range(HR_LIP, HR_ERR_CODE+1):
            set_hr(ctx, idx,0)
        cleanup(to_clean)
        print("[ORCH] Reset done. Back to IDLE.")

    threading.Thread(target=ack_wait_thread, daemon=True).start()

# ---------- Threaded watcher ----------
def inference_thread(ctx: ModbusServerContext, args):
    while True:
        if get_hr(ctx, HR_START,1)[0]==1:
            do_inference_cycle(ctx,args)
        time.sleep(0.001)  # check very fast for PLC START

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Fast Modbus Orchestrator")
    ap.add_argument("--bind-ip", default=BIND_IP)
    ap.add_argument("--bind-port", type=int, default=BIND_PORT)
    ap.add_argument("--timeout", type=float, default=RESULT_TIMEOUT_S)
    ap.add_argument("--ack-timeout", type=float, default=ACK_TIMEOUT_S)
    ap.add_argument("--lip-trigger", default=LIP_TRIGGER_DEFAULT)
    ap.add_argument("--lip-result", default=LIP_RESULT_DEFAULT)
    ap.add_argument("--body-trigger", default=BODY_TRIGGER_DEFAULT)
    ap.add_argument("--body-result", default=BODY_RESULT_DEFAULT)
    ap.add_argument("--bodycl-trigger", default=BODYCL_TRIGGER_DEFAULT)
    ap.add_argument("--bodycl-result", default=BODYCL_RESULT_DEFAULT)
    args = ap.parse_args()

    hr_block = LoggingDataBlock(0,[0]*32,"HR")
    di_block = LoggingDataBlock(0,[0]*1,"DI")
    co_block = LoggingDataBlock(0,[0]*1,"COIL")
    ir_block = LoggingDataBlock(0,[0]*1,"IR")

    def on_write(name,address,values):
        if name=="HR":
            print(f"[PLC->MODBUS] WRITE {name} addr={address} values={values}")
    hr_block.on_write = on_write

    store = ModbusSlaveContext(di=di_block,co=co_block,hr=hr_block,ir=ir_block)
    context = ModbusServerContext(slaves=store,single=True)

    # init HRs
    for idx in range(0,9): set_hr(context, idx,0)

    start_modbus_server(context,args.bind_ip,args.bind_port)
    print(f"[SERVER] Modbus TCP listening on {args.bind_ip}:{args.bind_port}")

    q = Queue()
    threading.Thread(target=keyboard_thread,args=(q,),daemon=True).start()
    threading.Thread(target=inference_thread,args=(context,args),daemon=True).start()

    try:
        while True:
            try:
                cmd = q.get_nowait()
                if cmd=="q": break
                elif cmd=="1": set_hr(context, HR_START,1)
                elif cmd=="r": set_hr(context, HR_ACK,1)
            except Empty: pass
            time.sleep(0.001)  # very fast polling
    except KeyboardInterrupt:
        print("\n[ORCH] CTRL-C -> exit.")
    finally:
        print("[ORCH] Bye.")

if __name__=="__main__":
    main()

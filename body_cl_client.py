# -- coding: utf-8 --
import os, sys, time
from ctypes import *
from typing import List, Optional
import argparse
import numpy as np
import cv2

# ====== HIK SDK bindings ======
sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *

# ====== Roboflow Inference ======
from inference_sdk import InferenceHTTPClient

# ---------- CONFIG DI DEFAULT ----------
DEFAULT_API_URL  = "http://localhost:9001"
DEFAULT_API_KEY  = "EC9puzE6crcRm7buAF1S"   # <-- sostituisci se necessario
# Per ora riutilizziamo lo stesso modello/soglia del lip_client come richiesto
DEFAULT_MODEL_ID = "big400-lip-insp-before-cleaning-da2vm/4"
DEFAULT_TH       = 0.18                     # soglia confidenza per contare difetto
DURATION_S       = 6.0                      # finestra temporale
INFER_MAX_W      = 1280                     # ridimensionamento (aspect-preserving)
INFER_EVERY_N    = 1                        # 1 = inferenza ad ogni frame

# ---------- UI ----------
WIN_NAME    = "Body-CL Inspection - HIK + Roboflow (q per uscire)"
PIXEL_MONO8 = 0x01080001
PREFIX      = "body_cl_client"

def clamp(v, vmin, vmax): return max(vmin, min(v, vmax))

def resize_keep_aspect(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    scale = max_w / float(w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA), scale

def frame_to_bgr(stOutFrame):
    info = stOutFrame.stFrameInfo
    w, h = info.nWidth, info.nHeight
    pt = info.enPixelType
    buf = string_at(stOutFrame.pBufAddr, info.nFrameLen)
    if pt == PIXEL_MONO8:
        img = np.frombuffer(buf, dtype=np.uint8, count=w*h).reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"PixelType non gestito: 0x{pt:x}. Imposta Mono8 o aggiungi demosaic.")

def normalize_conf(c):
    """Supporta conf in [0..1] o [0..100]."""
    try:
        conf = float(c)
    except Exception:
        return 0.0
    return conf/100.0 if conf > 1.0 else conf

def draw_info(img, txt):
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def draw_predictions(img, preds, th: float, ok_classes: Optional[List[str]] = None):
    """Disegna box SOLO per predizioni con confidenza >= th.
       Verde se la classe è in whitelist (logo/logo_front), rosso altrimenti.
    """
    if not preds or "predictions" not in preds:
        return
    wl = set([c.strip().lower() for c in (ok_classes or []) if c.strip()]) if ok_classes else set()
    h, w = img.shape[:2]

    for p in preds["predictions"]:
        conf = normalize_conf(p.get("confidence", 0.0))
        if conf < th:
            continue  # non disegnare sotto soglia

        x = p.get("x"); y = p.get("y")
        ww = p.get("width"); hh = p.get("height")
        cls = p.get("class", "obj")
        if None in (x, y, ww, hh):
            continue

        x_min = int(x - ww/2); y_min = int(y - hh/2)
        x_max = int(x + ww/2); y_max = int(y + hh/2)
        x_min = clamp(x_min, 0, w-1); y_min = clamp(y_min, 0, h-1)
        x_max = clamp(x_max, 0, w-1); y_max = clamp(y_max, 0, h-1)

        cls_norm = str(cls).lower().replace(" ", "")
        color = (0, 255, 0) if cls_norm in wl else (0, 0, 255)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img, f"{cls} {conf:.2f}",
                    (x_min, max(0, y_min-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def any_defect(preds, th: float, ok_classes: Optional[List[str]], debug: bool = False):
    """True se c'è almeno una predizione (conf≥th) NON in whitelist."""
    if not preds or "predictions" not in preds:
        return False
    wl = set([c.strip().lower() for c in (ok_classes or []) if c.strip()])

    for p in preds["predictions"]:
        conf = normalize_conf(p.get("confidence", 0.0))
        cls  = str(p.get("class", "")).strip().lower()
        if debug:
            print(f"[debug] class={cls} conf={conf:.3f} th={th:.3f} in_wl={cls in wl}", file=sys.stderr)
        if conf >= th and (cls not in wl):
            return True
    return False

# ====== HIK helpers ======
def hik_open_by_index(idx: int):
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        raise RuntimeError(f"Enum devices fail or no device! ret=0x{ret:x}")
    if idx >= deviceList.nDeviceNum:
        raise RuntimeError(f"Device index {idx} out of range (found {deviceList.nDeviceNum})")
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
    if cam.MV_CC_CreateHandle(stDeviceList) != 0:
        raise RuntimeError("Create handle fail")
    if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
        raise RuntimeError("Open device fail (già aperta?)")
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetEnumValue("PixelFormat", PIXEL_MONO8)
    try:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 25.0)
    except: pass
    if cam.MV_CC_StartGrabbing() != 0:
        cam.MV_CC_CloseDevice(); cam.MV_CC_DestroyHandle()
        raise RuntimeError("Start grabbing fail")
    return cam

def hik_grab_one(cam, timeout_ms=800):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    ret = cam.MV_CC_GetImageBuffer(stOutFrame, timeout_ms)
    if ret != 0:
        return None
    try:
        return frame_to_bgr(stOutFrame)
    finally:
        cam.MV_CC_FreeImageBuffer(stOutFrame)

def hik_close(cam):
    try: cam.MV_CC_StopGrabbing()
    except: pass
    try: cam.MV_CC_CloseDevice()
    except: pass
    try: cam.MV_CC_DestroyHandle()
    except: pass

# ====== ARGPARSE ======
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-index", type=int, default=1, help="Indice HIK (default 2 per Body-CL)")
    ap.add_argument("--api-url", default=os.getenv("RF_API_URL", DEFAULT_API_URL))
    ap.add_argument("--api-key", default=os.getenv("RF_API_KEY", DEFAULT_API_KEY))
    ap.add_argument("--model",   default=os.getenv("RF_MODEL_ID", DEFAULT_MODEL_ID))
    ap.add_argument("--th", type=float, default=float(os.getenv("RF_TH", DEFAULT_TH)))
    ap.add_argument("--ok-classes", default="logo,logo_front", help="CSV classi da NON considerare difetto (whitelist)")
    ap.add_argument("--duration", type=float, default=DURATION_S, help="Durata finestra (s)")
    ap.add_argument("--save-last", default=None, help="Se settato, salva ultimo frame annotato (JPG)")
    ap.add_argument("--debug", action="store_true", help="Logga classi/confidenze su stderr")

    # modalità daemon (finestre sempre aperte + trigger/result file)
    ap.add_argument("--daemon", action="store_true",
                    help="Keep camera+window open; wait for trigger-file to run a 6s inference")
    ap.add_argument("--trigger-file", default="/tmp/bodycl_go",
                    help="Path to trigger file (presence triggers one inference window)")
    ap.add_argument("--result-file",  default="/tmp/bodycl_res",
                    help="Path to result file (token written here)")
    return ap.parse_args()

# ====== MAIN ======
def main():
    args = parse_args()
    ok_classes = [s.strip().lower() for s in args.ok_classes.split(",")] if args.ok_classes else []

    # Client Roboflow
    client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)

    # HIK init
    MvCamera.MV_CC_Initialize()
    try:
        cam = hik_open_by_index(args.cam_index)
    except Exception as e:
        print(f"[{PREFIX}] open error: {e}", file=sys.stderr)
        MvCamera.MV_CC_Finalize()
        sys.exit(2)

    try:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 800)

        if not args.daemon:
            # ====== Modalità ONE-SHOT: 6s e termina ======
            t0 = time.time()
            frame_id = 0
            found_defect = False
            last_view = None

            while (time.time() - t0) < args.duration:
                img = hik_grab_one(cam, timeout_ms=800)
                if img is None:
                    if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                    continue

                frame_id += 1
                work_img, _ = resize_keep_aspect(img, INFER_MAX_W)

                preds = None
                if (frame_id % INFER_EVERY_N) == 0:
                    try:
                        preds = client.infer(work_img, model_id=args.model)
                    except Exception as e:
                        print(f"[{PREFIX}] inference error: {e}", file=sys.stderr)

                view = work_img.copy()
                if preds:
                    draw_predictions(view, preds, th=args.th, ok_classes=ok_classes)
                    if any_defect(preds, th=args.th, ok_classes=ok_classes, debug=args.debug):
                        found_defect = True

                draw_info(view, f"[{int((time.time()-t0)*1000)}ms/{int(args.duration*1000)}ms] TH={args.th}")
                cv2.imshow(WIN_NAME, view)
                last_view = view
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if args.save_last and last_view is not None:
                try: cv2.imwrite(args.save_last, last_view)
                except Exception as e:
                    print(f"[{PREFIX}] save_last error: {e}", file=sys.stderr)

            print("BODY_CL_NOK" if found_defect else "BODY_CL_OK")
            sys.exit(0)

        else:
            # ====== Modalità DAEMON ======
            print(f"[{PREFIX}] DAEMON mode. Trigger='{args.trigger_file}'  Result='{args.result_file}'")
            while True:
                img = hik_grab_one(cam, timeout_ms=800)
                if img is not None:
                    view, _ = resize_keep_aspect(img, INFER_MAX_W)
                    draw_info(view, "[IDLE] waiting trigger")
                    cv2.imshow(WIN_NAME, view)

                # trigger-file presente -> esegui finestra 6s
                if os.path.exists(args.trigger_file):
                    try: os.remove(args.trigger_file)
                    except: pass

                    t0 = time.time()
                    frame_id = 0
                    found_defect = False
                    last_view = None
                    while (time.time() - t0) < args.duration:
                        img = hik_grab_one(cam, timeout_ms=800)
                        if img is None:
                            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
                            continue

                        frame_id += 1
                        work_img, _ = resize_keep_aspect(img, INFER_MAX_W)

                        preds = None
                        if (frame_id % INFER_EVERY_N) == 0:
                            try:
                                preds = client.infer(work_img, model_id=args.model)
                            except Exception as e:
                                print(f"[{PREFIX}] inference error: {e}", file=sys.stderr)

                        view = work_img.copy()
                        if preds:
                            draw_predictions(view, preds, th=args.th, ok_classes=ok_classes)
                            if any_defect(preds, th=args.th, ok_classes=ok_classes, debug=args.debug):
                                found_defect = True

                        draw_info(view, f"[RUN {int(time.time()-t0)}s] TH={args.th}")
                        cv2.imshow(WIN_NAME, view)
                        last_view = view
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            break

                    token = "BODY_CL_NOK" if found_defect else "BODY_CL_OK"
                    print(token)  # anche su stdout
                    try:
                        with open(args.result_file, "w") as f:
                            f.write(token + "\n")
                    except Exception as e:
                        print(f"[{PREFIX}] write result-file error: {e}", file=sys.stderr)

                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

    finally:
        try: cv2.destroyAllWindows()
        except: pass
        try: hik_close(cam)
        except: pass
        try: MvCamera.MV_CC_Finalize()
        except: pass

if __name__ == "__main__":
    main()

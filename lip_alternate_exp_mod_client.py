# -- coding: utf-8 --
import os, sys, time, argparse
from ctypes import *
from typing import Optional, List
import numpy as np
import cv2

# ====== HIK SDK bindings ======
sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *

# ====== Roboflow Inference ======
from inference_sdk import InferenceHTTPClient

# ---------- DEFAULT CONFIG ----------
DEFAULT_API_URL  = "http://localhost:9001"
DEFAULT_API_KEY  = "EC9puzE6crcRm7buAF1S"

# Modelli + soglie per exposure
DEFAULT_MODEL_HIGH = "big400-lip-insp-before-cleaning-da2vm/4"
DEFAULT_TH_HIGH    = 0.18    # conf threshold per expo alta
DEFAULT_MODEL_LOW  = "big400-body-insp-before-cleaning-7cr8v/3"
DEFAULT_TH_LOW     = 0.31    # conf threshold per expo bassa

# Acquisizione / inferenza
DEFAULT_FPS          = 20.0   # configurabile con --fps
DEFAULT_INFER_EVERYN = 1      # ogni quanti frame inferire (default: 1)
INFER_MAX_W          = 1280

# Exposure alternato
DEFAULT_EXPO_HIGH_US = 5000.0   # µs
DEFAULT_EXPO_LOW_US  = 500.0    # µs
DEFAULT_SWITCH_SEC   = 1.0      # alterna ogni 1s

# Finestra one-shot
DURATION_S = 6.0

# UI
WIN_NAME    = "LIP Alt-Exposure Client (q per uscire)"
PIXEL_MONO8 = 0x01080001
PREFIX      = "lip_alt_client"

# -------- utils --------
def clamp(v, vmin, vmax): return max(vmin, min(v, vmax))

def resize_keep_aspect(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    s = max_w / float(w)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA), s

def normalize_conf(c):
    try:
        conf = float(c)
    except Exception:
        return 0.0
    return conf/100.0 if conf > 1.0 else conf

def draw_info(img, txt):
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def draw_predictions(img, preds, th: float, ok_classes: Optional[List[str]] = None):
    """Box verdi per whitelist (logo, logo_front), rossi per il resto, solo conf >= th."""
    if not preds or "predictions" not in preds:
        return
    wl = set([c.strip().lower() for c in (ok_classes or []) if c.strip()]) if ok_classes else set()
    h, w = img.shape[:2]
    for p in preds["predictions"]:
        conf = normalize_conf(p.get("confidence", 0.0))
        if conf < th:
            continue
        x = p.get("x"); y = p.get("y")
        ww = p.get("width"); hh = p.get("height")
        cls = str(p.get("class", "obj"))
        if None in (x, y, ww, hh):
            continue
        x_min = int(x - ww/2); y_min = int(y - hh/2)
        x_max = int(x + ww/2); y_max = int(y + hh/2)
        x_min = clamp(x_min, 0, w-1); y_min = clamp(y_min, 0, h-1)
        x_max = clamp(x_max, 0, w-1); y_max = clamp(y_max, 0, h-1)
        cls_norm = cls.lower().replace(" ", "")
        color = (0,255,0) if cls_norm in wl else (0,0,255)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (x_min, max(0, y_min-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def frame_to_bgr(stOutFrame):
    info = stOutFrame.stFrameInfo
    w, h = info.nWidth, info.nHeight
    pt = info.enPixelType
    buf = string_at(stOutFrame.pBufAddr, info.nFrameLen)
    if pt == PIXEL_MONO8:
        img = np.frombuffer(buf, dtype=np.uint8, count=w*h).reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"PixelType non gestito: 0x{pt:x}. Imposta Mono8 o aggiungi demosaic.")

# -------- HIK helpers --------
def set_exposure_us(cam, us):
    try: cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # OFF
    except: pass
    cam.MV_CC_SetFloatValue("ExposureTime", float(us))

def hik_open_by_index(idx: int, fps: float):
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

    # imposta FPS richiesto
    try:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
    except:
        pass

    if cam.MV_CC_StartGrabbing() != 0:
        cam.MV_CC_CloseDevice(); cam.MV_CC_DestroyHandle()
        raise RuntimeError("Start grabbing fail")
    return cam

def hik_grab_one(cam, timeout_ms=800):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(MV_FRAME_OUT))
    ret = cam.MV_CC_GetImageBuffer(stOutFrame, timeout_ms)
    if ret != 0:
        return None, None
    try:
        img = frame_to_bgr(stOutFrame)
        return img, stOutFrame
    except:
        cam.MV_CC_FreeImageBuffer(stOutFrame)
        raise

def hik_release_frame(cam, stOutFrame):
    try: cam.MV_CC_FreeImageBuffer(stOutFrame)
    except: pass

def hik_close(cam):
    try: cam.MV_CC_StopGrabbing()
    except: pass
    try: cam.MV_CC_CloseDevice()
    except: pass
    try: cam.MV_CC_DestroyHandle()
    except: pass

# -------- Argparse --------
def parse_args():
    ap = argparse.ArgumentParser(description="LIP alternate exposure client (one-shot or daemon)")
    ap.add_argument("--cam-index", type=int, default=0, help="Indice HIK (default 0)")

    # Inference endpoints
    ap.add_argument("--api-url", default=os.getenv("RF_API_URL", DEFAULT_API_URL))
    ap.add_argument("--api-key", default=os.getenv("RF_API_KEY", DEFAULT_API_KEY))

    # Modelli/soglie per exposure
    ap.add_argument("--model-high", default=os.getenv("RF_MODEL_HIGH", DEFAULT_MODEL_HIGH))
    ap.add_argument("--th-high", type=float, default=float(os.getenv("RF_TH_HIGH", DEFAULT_TH_HIGH)))
    ap.add_argument("--model-low",  default=os.getenv("RF_MODEL_LOW",  DEFAULT_MODEL_LOW))
    ap.add_argument("--th-low",  type=float, default=float(os.getenv("RF_TH_LOW",  DEFAULT_TH_LOW)))

    # FPS / frequenza inferenza
    ap.add_argument("--fps", type=float, default=float(os.getenv("RF_FPS", DEFAULT_FPS)),
                    help="Frame rate richiesto alla camera")
    ap.add_argument("--infer-every-n", type=int, default=int(os.getenv("RF_INFER_EVERY_N", DEFAULT_INFER_EVERYN)),
                    help="Inferenza ogni N frame (default 1)")

    # Exposure toggling
    ap.add_argument("--expo-high-us", type=float, default=DEFAULT_EXPO_HIGH_US)
    ap.add_argument("--expo-low-us",  type=float, default=DEFAULT_EXPO_LOW_US)
    ap.add_argument("--switch-period", type=float, default=DEFAULT_SWITCH_SEC,
                    help="Periodo (s) di alternanza exposure")

    # One-shot window
    ap.add_argument("--duration", type=float, default=DURATION_S, help="Durata finestra (s)")
    ap.add_argument("--save-last", default=None, help="Se settato, salva ultimo frame annotato (JPG)")
    ap.add_argument("--ok-classes", default="logo,logo_front",
                    help="CSV classi in whitelist (box verdi)")

    # Daemon mode + orchestrator paths
    ap.add_argument("--daemon", action="store_true",
                    help="Keep camera+window open; wait for trigger-file to run a window")
    ap.add_argument("--trigger-file", default="/tmp/lip_go",
                    help="Path trigger file")
    ap.add_argument("--result-file",  default="/tmp/lip_res",
                    help="Path result file")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

# -------- Core run (window loop for a fixed duration) --------
def run_window(cam, client, args) -> (bool, np.ndarray):
    """
    Ritorna: (found_defect, last_view)
    Alterna exposure ogni args.switch_period e inferisce ogni args.infer_every_n frame;
    usa modello/soglia in base a exposure corrente.
    """
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 800)

    ok_classes = [s.strip().lower() for s in args.ok_classes.split(",")] if args.ok_classes else []
    current_high = True
    set_exposure_us(cam, args.expo_high_us)
    last_switch = time.time()

    t0 = time.time()
    frame_id = 0
    found_defect = False
    last_view = None

    while (time.time() - t0) < args.duration:
        # alterna exposure
        now = time.time()
        if (now - last_switch) >= args.switch_period:
            current_high = not current_high
            set_exposure_us(cam, args.expo_high_us if current_high else args.expo_low_us)
            last_switch = now

        img, st = hik_grab_one(cam, timeout_ms=800)
        if img is None:
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            continue

        try:
            frame_id += 1
            work_img, _ = resize_keep_aspect(img, INFER_MAX_W)

            preds = None
            if (frame_id % max(1, args.infer_every_n)) == 0:
                model = args.model_high if current_high else args.model_low
                th    = args.th_high    if current_high else args.th_low
                try:
                    preds = client.infer(work_img, model_id=model)
                except Exception as e:
                    if args.debug: print(f"[{PREFIX}] inference error: {e}", file=sys.stderr)

            view = work_img.copy()
            if preds:
                th = args.th_high if current_high else args.th_low
                draw_predictions(view, preds, th=th, ok_classes=ok_classes)
                # consideriamo “difetto” qualsiasi predizione >= th NON in whitelist
                if preds and "predictions" in preds:
                    for p in preds["predictions"]:
                        conf = normalize_conf(p.get("confidence",0.0))
                        cls  = str(p.get("class","")).strip().lower()
                        if conf >= (args.th_high if current_high else args.th_low) and cls not in ok_classes:
                            found_defect = True
                            break

            label = f"EXP={'HIGH' if current_high else 'LOW'} ({int(args.expo_high_us if current_high else args.expo_low_us)} µs) | FPS={args.fps:.1f} | N={args.infer_every_n}"
            draw_info(view, label)
            cv2.imshow(WIN_NAME, view)
            last_view = view

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        finally:
            hik_release_frame(cam, st)

    return found_defect, last_view

# -------- MAIN --------
def main():
    args = parse_args()
    client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)

    # avviso se exposure > periodo frame
    try:
        period_us = 1e6 / float(args.fps)
        if args.expo_high_us > period_us or args.expo_low_us > period_us:
            print(f"[WARN] Exposure ({args.expo_high_us}/{args.expo_low_us} µs) supera il frame period ~{int(period_us)} µs @ {args.fps:.1f} FPS. "
                  "Riduci exposure o FPS per evitare motion blur/drop.", file=sys.stderr)
    except:
        pass

    MvCamera.MV_CC_Initialize()
    try:
        cam = hik_open_by_index(args.cam_index, fps=args.fps)
    except Exception as e:
        print(f"[{PREFIX}] open error: {e}", file=sys.stderr)
        MvCamera.MV_CC_Finalize()
        sys.exit(2)

    try:
        if not args.daemon:
            # ---------- ONE SHOT ----------
            found_defect, last_view = run_window(cam, client, args)
            if args.save_last and last_view is not None:
                try: cv2.imwrite(args.save_last, last_view)
                except Exception as e:
                    print(f"[{PREFIX}] save_last error: {e}", file=sys.stderr)
            print("LIP_NOK" if found_defect else "LIP_OK")
            sys.exit(0)

        else:
            # ---------- DAEMON ----------
            print(f"[{PREFIX}] DAEMON mode. Trigger='{args.trigger_file}'  Result='{args.result_file}'")
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_NAME, 1280, 800)

            # preview idle con exposure che continua ad alternare
            current_high = True
            set_exposure_us(cam, args.expo_high_us)
            last_switch = time.time()

            while True:
                # alterna exposure anche in idle
                now = time.time()
                if (now - last_switch) >= args.switch_period:
                    current_high = not current_high
                    set_exposure_us(cam, args.expo_high_us if current_high else args.expo_low_us)
                    last_switch = now

                img, st = hik_grab_one(cam, timeout_ms=800)
                if img is not None:
                    view, _ = resize_keep_aspect(img, INFER_MAX_W)
                    draw_info(view, f"[IDLE] EXP={'HIGH' if current_high else 'LOW'} ({int(args.expo_high_us if current_high else args.expo_low_us)} µs)")
                    cv2.imshow(WIN_NAME, view)
                    hik_release_frame(cam, st)

                # trigger?
                if os.path.exists(args.trigger_file):
                    try: os.remove(args.trigger_file)
                    except: pass
                    found_defect, last_view = run_window(cam, client, args)
                    token = "LIP_NOK" if found_defect else "LIP_OK"
                    print(token)
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

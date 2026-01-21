# -- coding: utf-8 --
"""
lip_client.py

Lip inspection client (HIK Mono8 + Roboflow) with:
- daemon trigger mode (--daemon, --trigger-file, --result-file)
- optional AWS S3 upload of NOK images (raw + annotated) with tags
- NEW: headless mode (--headless) to run without any OpenCV window / imshow

Production intent:
- Run headless + daemon for max efficiency.
- Save only NOK frames to S3 by enabling --save-on-defect.
- One upload per RUN window (first defect) to avoid flooding S3.
- Optional --force-upload for debug.

S3 layout:
s3://<bucket>/<site>/<station>/{raw|annotated}/<UTC_TS>_<EVENT>.jpg
"""

import os, sys, time
from ctypes import *
from typing import List, Optional, Deque, Tuple
import argparse
import numpy as np
import cv2
from collections import deque

import uuid
from datetime import datetime, timezone

# ====== AWS S3 ======
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ====== HIK SDK bindings ======
sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *

# ====== Roboflow Inference ======
from inference_sdk import InferenceHTTPClient


# ---------- CONFIG DI DEFAULT ----------
DEFAULT_API_URL  = "http://localhost:9001"
DEFAULT_API_KEY  = "EC9puzE6crcRm7buAF1S"
DEFAULT_MODEL_ID = "big400-lip-insp-before-cleaning-da2vm/9"
DEFAULT_TH       = 0.10
DURATION_S       = 3.0
INFER_MAX_W      = 1280
INFER_EVERY_N    = 1

DEFAULT_FPS      = 25.0
DEFAULT_EXPO_US  = 10000.0
DEFAULT_GAIN     = 0.0

# ---------- AWS defaults ----------
DEFAULT_BUCKET   = "bewtr-bottle-defect-validation"
DEFAULT_SITE     = "nice"
DEFAULT_STATION  = "lip"

# ---------- UI ----------
WIN_NAME    = "Lip Inspection - HIK + Roboflow (q per uscire)"
PIXEL_MONO8 = 0x01080001
PREFIX      = "lip_client"


def clamp(v, vmin, vmax): return max(vmin, min(v, vmax))


def resize_keep_aspect(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    scale = max_w / float(w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale


def frame_to_bgr(stOutFrame):
    info = stOutFrame.stFrameInfo
    w, h = info.nWidth, info.nHeight
    pt = info.enPixelType
    buf = string_at(stOutFrame.pBufAddr, info.nFrameLen)
    if pt == PIXEL_MONO8:
        img = np.frombuffer(buf, dtype=np.uint8, count=w * h).reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise RuntimeError(f"PixelType non gestito: 0x{pt:x}. Imposta Mono8 o aggiungi demosaic.")


def normalize_conf(c):
    try:
        conf = float(c)
    except Exception:
        return 0.0
    return conf / 100.0 if conf > 1.0 else conf


# ---------- HUD helpers ----------
class FPSMeter:
    """FPS misurati su finestra mobile (default: 30 frame)"""
    def __init__(self, window: int = 30):
        self.ts: Deque[float] = deque(maxlen=max(5, int(window)))

    def tick(self) -> float:
        now = time.time()
        self.ts.append(now)
        if len(self.ts) < 2:
            return 0.0
        dt = self.ts[-1] - self.ts[0]
        if dt <= 1e-9:
            return 0.0
        return (len(self.ts) - 1) / dt


def draw_hud(img, lines: List[str], origin=(12, 18), line_h=22,
             font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.55, thickness=1):
    """Disegna più righe con background semitrasparente."""
    if not lines:
        return

    x, y = origin
    sizes: List[Tuple[int, int]] = []
    max_w = 0
    for t in lines:
        (tw, th), _ = cv2.getTextSize(t, font, scale, thickness)
        sizes.append((tw, th))
        max_w = max(max_w, tw)

    box_w = max_w + 16
    box_h = len(lines) * line_h + 12

    overlay = img.copy()
    cv2.rectangle(overlay, (x - 6, y - 14), (x - 6 + box_w, y - 14 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    yy = y
    for t in lines:
        cv2.putText(img, t, (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        yy += line_h


def draw_predictions(img, preds, th: float, ok_classes: Optional[List[str]] = None):
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
        cls = p.get("class", "obj")
        if None in (x, y, ww, hh):
            continue
        x_min = int(x - ww / 2); y_min = int(y - hh / 2)
        x_max = int(x + ww / 2); y_max = int(y + hh / 2)
        x_min = clamp(x_min, 0, w - 1); y_min = clamp(y_min, 0, h - 1)
        x_max = clamp(x_max, 0, w - 1); y_max = clamp(y_max, 0, h - 1)
        cls_norm = str(cls).lower().replace(" ", "")
        color = (0, 255, 0) if cls_norm in wl else (0, 0, 255)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (x_min, max(0, y_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def any_defect(preds, th: float, ok_classes: Optional[List[str]], debug: bool = False):
    """True se c'è almeno una predizione (conf ≥ th) NON in whitelist."""
    if not preds or "predictions" not in preds:
        return False
    wl = set([c.strip().lower() for c in (ok_classes or []) if c.strip()])
    for p in preds["predictions"]:
        conf = normalize_conf(p.get("confidence", 0.0))
        cls = str(p.get("class", "")).strip().lower()
        if debug:
            print(f"[debug] class={cls} conf={conf:.3f} th={th:.3f} in_wl={cls in wl}", file=sys.stderr)
        if conf >= th and (cls not in wl):
            return True
    return False


# ====== HIK helpers: Exposure & Gain ======
def set_exposure_us(cam, us):
    try:
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # 0=Off
    except:
        pass
    cam.MV_CC_SetFloatValue("ExposureTime", float(us))


def get_exposure_us(cam):
    try:
        val = MVCC_FLOATVALUE()
        memset(byref(val), 0, sizeof(MVCC_FLOATVALUE))
        if cam.MV_CC_GetFloatValue("ExposureTime", val) == 0:
            return val.fCurValue
    except:
        pass
    return None


def set_gain(cam, gain_val):
    try:
        cam.MV_CC_SetEnumValue("GainAuto", 0)  # 0=Off
    except:
        pass
    cam.MV_CC_SetFloatValue("Gain", float(gain_val))


def get_gain(cam):
    try:
        val = MVCC_FLOATVALUE()
        memset(byref(val), 0, sizeof(MVCC_FLOATVALUE))
        if cam.MV_CC_GetFloatValue("Gain", val) == 0:
            return val.fCurValue
    except:
        pass
    return None


def hik_open_by_index(idx: int, fps: float, exposure_us: float, gain_val: float):
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

    # FPS richiesto
    try:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
    except:
        pass

    set_exposure_us(cam, exposure_us)
    set_gain(cam, gain_val)

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


# ====== AWS S3 helpers ======
def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def s3_put_jpg(s3_client, bucket: str, key: str, bgr_img: np.ndarray, extra_tags: Optional[dict] = None):
    ok, enc = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    body = enc.tobytes()

    tag_str = None
    if extra_tags:
        def esc(x): return str(x).replace(" ", "_")
        tag_str = "&".join([f"{esc(k)}={esc(v)}" for k, v in extra_tags.items()])

    kwargs = dict(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="image/jpeg",
    )
    if tag_str:
        kwargs["Tagging"] = tag_str

    s3_client.put_object(**kwargs)


# ====== ARGPARSE ======
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cam-index", type=int, default=2, help="Indice HIK (default 2)")
    ap.add_argument("--api-url", default=os.getenv("RF_API_URL", DEFAULT_API_URL))
    ap.add_argument("--api-key", default=os.getenv("RF_API_KEY", DEFAULT_API_KEY))
    ap.add_argument("--model",   default=os.getenv("RF_MODEL_ID", DEFAULT_MODEL_ID))
    ap.add_argument("--th", type=float, default=float(os.getenv("RF_TH", DEFAULT_TH)))
    ap.add_argument("--ok-classes", default="logo,logo_front",
                    help="CSV classi da NON considerare difetto (whitelist)")
    ap.add_argument("--duration", type=float, default=DURATION_S, help="Durata finestra (s)")
    ap.add_argument("--save-last", default=None, help="Se settato, salva ultimo frame annotato (JPG)")
    ap.add_argument("--debug", action="store_true", help="Logga classi/confidenze su stderr")

    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help="Frame rate richiesto alla camera (default 25.0)")
    ap.add_argument("--exposure-us", type=float, default=DEFAULT_EXPO_US,
                    help="Exposure time fisso in microsecondi (default 10000.0)")
    ap.add_argument("--gain", type=float, default=DEFAULT_GAIN,
                    help="Gain manuale (tipicamente dB su HIK). Default 0.0")

    ap.add_argument("--daemon", action="store_true",
                    help="Keep camera open; wait for trigger-file to run inference window")
    ap.add_argument("--trigger-file", default="/tmp/lip_go",
                    help="Path to trigger file (presence triggers one inference window)")
    ap.add_argument("--result-file",  default="/tmp/lip_res",
                    help="Path to result file (token written here)")

    # NEW: headless (no windows / no imshow / no waitKey)
    ap.add_argument("--headless", action="store_true",
                    help="Run without any OpenCV GUI (no cv2.namedWindow/imshow). Best for production.")

    # --- AWS / validation ---
    ap.add_argument("--bucket", default=os.getenv("AWS_S3_BUCKET", DEFAULT_BUCKET),
                    help=f"S3 bucket per validation (default {DEFAULT_BUCKET})")
    ap.add_argument("--site", default=os.getenv("PLANT_SITE", DEFAULT_SITE),
                    help="Plant/site, es: nice, tianjin, bern")
    ap.add_argument("--station", default=os.getenv("CAM_STATION", DEFAULT_STATION),
                    help="Stazione/camera: lip, body, body_cl")
    ap.add_argument("--save-on-defect", action="store_true", default=False,
                    help="Se presente, salva raw+annotated su S3 SOLO quando detecta difetto")
    ap.add_argument("--force-upload", action="store_true",
                    help="DEBUG: forza upload su S3 al primo frame della finestra RUN (anche se OK o inference error)")

    return ap.parse_args()


def fmt_expo_gain(cam, args):
    expo_now = get_exposure_us(cam)
    gain_now = get_gain(cam)
    expo = int(expo_now) if expo_now is not None else int(args.exposure_us)
    gain = float(gain_now) if gain_now is not None else float(args.gain)
    return expo, gain


def safe_imshow(headless: bool, win: str, img: np.ndarray):
    if headless:
        return
    cv2.imshow(win, img)


def safe_waitkey(headless: bool, delay_ms: int = 1) -> int:
    if headless:
        return -1
    return cv2.waitKey(delay_ms) & 0xFF


def main():
    args = parse_args()

    ok_classes = [s.strip().lower() for s in args.ok_classes.split(",")] if args.ok_classes else []
    client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)

    # AWS client (usa ~/.aws/credentials configurato con aws configure)
    s3 = boto3.client("s3", region_name="eu-north-1")

    print(f"[{PREFIX}] headless={args.headless} save_on_defect={args.save_on_defect} force_upload={args.force_upload} "
          f"bucket={args.bucket} site={args.site} station={args.station}", file=sys.stderr)

    # Warning exposure > period
    try:
        period_us = 1e6 / float(args.fps)
        if args.exposure_us > period_us:
            print(f"[WARN] Exposure ({args.exposure_us} µs) > frame period ~{int(period_us)} µs @ {args.fps:.1f} FPS.",
                  file=sys.stderr)
    except:
        pass

    MvCamera.MV_CC_Initialize()
    try:
        cam = hik_open_by_index(args.cam_index, fps=args.fps,
                                exposure_us=args.exposure_us, gain_val=args.gain)
    except Exception as e:
        print(f"[{PREFIX}] open error: {e}", file=sys.stderr)
        MvCamera.MV_CC_Finalize()
        sys.exit(2)

    fpsm = FPSMeter(window=30)
    last_infer_ms: Optional[float] = None

    def hud_lines(mode: str, run_s: float, fps_real: float, expo: int, gain: float,
                  th: float, model: str, infer_ms: Optional[float]):
        return [
            f"{mode}  runtime={run_s:5.2f}s  fps={fps_real:5.1f}",
            f"EXP={expo}us  GAIN={gain:.2f}  target_fps={args.fps:.1f}",
            f"model={model}",
            f"TH={th:.2f} (default={DEFAULT_TH:.2f})  infer_ms={infer_ms:.1f}" if infer_ms is not None else
            f"TH={th:.2f} (default={DEFAULT_TH:.2f})  infer_ms=--",
        ]

    def upload_pair_to_s3(site: str, station: str,
                          raw_img: np.ndarray, labeled_img: np.ndarray,
                          expo: int, gain: float, fps_real: float,
                          model: str, th: float, infer_ms: Optional[float]):
        ts = utc_now_str()
        ev = uuid.uuid4().hex[:12]

        base = f"{site}/{station}"
        raw_key = f"{base}/raw/{ts}_{ev}.jpg"
        lab_key = f"{base}/annotated/{ts}_{ev}.jpg"

        tags = {
            "site": site,
            "station": station,
            "model": model,
            "th": f"{th:.2f}",
            "expo_us": str(expo),
            "gain": f"{gain:.2f}",
        }
        if infer_ms is not None:
            tags["infer_ms"] = f"{infer_ms:.1f}"

        s3_put_jpg(s3, args.bucket, raw_key, raw_img, extra_tags=tags)
        s3_put_jpg(s3, args.bucket, lab_key, labeled_img, extra_tags=tags)

        print(f"[{PREFIX}] S3 saved:\n  RAW: s3://{args.bucket}/{raw_key}\n  LAB: s3://{args.bucket}/{lab_key}",
              file=sys.stderr)

    try:
        # GUI init only if not headless
        if not args.headless:
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_NAME, 1280, 800)

        if not args.daemon:
            # ====== ONE-SHOT ======
            t0 = time.time()
            frame_id = 0
            found_defect = False
            saved_this_run = False
            last_view = None

            while (time.time() - t0) < args.duration:
                img = hik_grab_one(cam, timeout_ms=800)
                if img is None:
                    if safe_waitkey(args.headless, 1) == ord('q'):
                        break
                    continue

                frame_id += 1
                fps_real = fpsm.tick()

                work_img, _ = resize_keep_aspect(img, INFER_MAX_W)

                preds = None
                if (frame_id % INFER_EVERY_N) == 0:
                    try:
                        t_inf0 = time.perf_counter()
                        preds = client.infer(work_img, model_id=args.model)
                        last_infer_ms = (time.perf_counter() - t_inf0) * 1000.0
                    except Exception as e:
                        print(f"[{PREFIX}] inference error: {e}", file=sys.stderr)

                raw_to_save = work_img.copy()
                view = work_img.copy()

                defect_now = False
                if preds:
                    draw_predictions(view, preds, th=args.th, ok_classes=ok_classes)
                    defect_now = any_defect(preds, th=args.th, ok_classes=ok_classes, debug=args.debug)
                    if defect_now:
                        found_defect = True

                expo, gain = fmt_expo_gain(cam, args)
                run_s = time.time() - t0

                # HUD only if not headless (saves CPU)
                if not args.headless:
                    draw_hud(view, hud_lines("RUN", run_s, fps_real, expo, gain, args.th, args.model, last_infer_ms))

                # Upload logic
                should_upload = False
                if args.save_on_defect and defect_now:
                    should_upload = True
                if args.force_upload:
                    should_upload = True

                if should_upload and not saved_this_run:
                    try:
                        print(f"[{PREFIX}] UPLOAD trigger (defect_now={defect_now}, force_upload={args.force_upload})",
                              file=sys.stderr)
                        upload_pair_to_s3(
                            site=args.site, station=args.station,
                            raw_img=raw_to_save, labeled_img=view,
                            expo=expo, gain=gain, fps_real=fps_real,
                            model=args.model, th=args.th, infer_ms=last_infer_ms
                        )
                        saved_this_run = True
                    except (BotoCoreError, ClientError, Exception) as e:
                        print(f"[{PREFIX}] S3 upload error: {e}", file=sys.stderr)

                safe_imshow(args.headless, WIN_NAME, view)
                last_view = view

                if safe_waitkey(args.headless, 1) == ord('q'):
                    break

            if args.save_last and last_view is not None:
                try:
                    cv2.imwrite(args.save_last, last_view)
                except Exception as e:
                    print(f"[{PREFIX}] save_last error: {e}", file=sys.stderr)

            print("LIP_NOK" if found_defect else "LIP_OK")
            sys.exit(0)

        else:
            # ====== DAEMON ======
            print(f"[{PREFIX}] DAEMON mode. Trigger='{args.trigger_file}'  Result='{args.result_file}'", file=sys.stderr)
            idle_t0 = time.time()

            while True:
                img = hik_grab_one(cam, timeout_ms=800)
                if img is not None:
                    fps_real = fpsm.tick()
                    view, _ = resize_keep_aspect(img, INFER_MAX_W)
                    expo, gain = fmt_expo_gain(cam, args)
                    run_s = time.time() - idle_t0

                    if not args.headless:
                        draw_hud(view, hud_lines("IDLE", run_s, fps_real, expo, gain, args.th, args.model, last_infer_ms))
                        safe_imshow(args.headless, WIN_NAME, view)

                # In headless mode: DO NOT call waitKey/imshow, just keep looping.

                # trigger -> finestra inference args.duration
                if os.path.exists(args.trigger_file):
                    try:
                        os.remove(args.trigger_file)
                    except:
                        pass

                    t0 = time.time()
                    frame_id = 0
                    found_defect = False
                    saved_this_run = False

                    # reset meter per fps più “puliti” in run
                    fpsm = FPSMeter(window=30)

                    while (time.time() - t0) < args.duration:
                        img = hik_grab_one(cam, timeout_ms=800)
                        if img is None:
                            if safe_waitkey(args.headless, 1) == ord('q'):
                                break
                            continue

                        frame_id += 1
                        fps_real = fpsm.tick()
                        work_img, _ = resize_keep_aspect(img, INFER_MAX_W)

                        preds = None
                        if (frame_id % INFER_EVERY_N) == 0:
                            try:
                                t_inf0 = time.perf_counter()
                                preds = client.infer(work_img, model_id=args.model)
                                last_infer_ms = (time.perf_counter() - t_inf0) * 1000.0
                            except Exception as e:
                                print(f"[{PREFIX}] inference error: {e}", file=sys.stderr)

                        raw_to_save = work_img.copy()
                        view = work_img.copy()

                        defect_now = False
                        if preds:
                            draw_predictions(view, preds, th=args.th, ok_classes=ok_classes)
                            defect_now = any_defect(preds, th=args.th, ok_classes=ok_classes, debug=args.debug)
                            if defect_now:
                                found_defect = True

                        expo, gain = fmt_expo_gain(cam, args)
                        run_s = time.time() - t0

                        if not args.headless:
                            draw_hud(view, hud_lines("RUN", run_s, fps_real, expo, gain, args.th, args.model, last_infer_ms))
                            safe_imshow(args.headless, WIN_NAME, view)

                        # Upload logic (one time per RUN)
                        should_upload = False
                        if args.save_on_defect and defect_now:
                            should_upload = True
                        if args.force_upload:
                            should_upload = True

                        if should_upload and not saved_this_run:
                            try:
                                print(f"[{PREFIX}] UPLOAD trigger (defect_now={defect_now}, force_upload={args.force_upload})",
                                      file=sys.stderr)
                                upload_pair_to_s3(
                                    site=args.site, station=args.station,
                                    raw_img=raw_to_save, labeled_img=view,
                                    expo=expo, gain=gain, fps_real=fps_real,
                                    model=args.model, th=args.th, infer_ms=last_infer_ms
                                )
                                saved_this_run = True
                            except (BotoCoreError, ClientError, Exception) as e:
                                print(f"[{PREFIX}] S3 upload error: {e}", file=sys.stderr)

                        if safe_waitkey(args.headless, 1) == ord('q'):
                            break

                    token = "LIP_NOK" if found_defect else "LIP_OK"
                    print(token)
                    try:
                        with open(args.result_file, "w") as f:
                            f.write(token + "\n")
                    except Exception as e:
                        print(f"[{PREFIX}] write result-file error: {e}", file=sys.stderr)

                    idle_t0 = time.time()
                    fpsm = FPSMeter(window=30)

                # allow 'q' quit only if not headless (since waitKey not called)
                if not args.headless and safe_waitkey(args.headless, 1) == ord('q'):
                    break

    finally:
        # GUI cleanup only if used
        if not args.headless:
            try: cv2.destroyAllWindows()
            except: pass
        try: hik_close(cam)
        except: pass
        try: MvCamera.MV_CC_Finalize()
        except: pass


if __name__ == "__main__":
    main()


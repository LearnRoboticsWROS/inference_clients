import threading
import queue
import cv2
import numpy as np
from ctypes import string_at

# Global exit flag
g_bExit = False
PIXEL_MONO8 = 0x01080001

# Display / zoom settings
USE_FIT = False
ZOOM = 0.7
ZOOM_MIN, ZOOM_MAX = 0.1, 2.0
WIN_NAME = "Hikrobot + Roboflow"

# Inference settings
INFER_EVERY_N_FRAMES = 2
INFER_MAX_W = 640  # smaller width → faster inference

# Shared queue for async inference
infer_queue = queue.Queue(maxsize=2)
last_preds_global = None

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def frame_to_bgr(stOutFrame):
    """Convert Mono8 buffer to BGR numpy"""
    info = stOutFrame.stFrameInfo
    w, h = info.nWidth, info.nHeight
    buf = string_at(stOutFrame.pBufAddr, info.nFrameLen)

    if info.enPixelType == PIXEL_MONO8:
        img = np.frombuffer(buf, dtype=np.uint8, count=w*h).reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"Unsupported PixelType: 0x{info.enPixelType:x}")

def resize_keep_aspect(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    scale = max_w / float(w)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA), scale

def draw_info(img, fps=None, mode_txt=""):
    if fps is not None:
        cv2.putText(img, f"{img.shape[1]}x{img.shape[0]}  FPS:{fps:.1f} {mode_txt}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def draw_predictions(img, preds):
    if not preds or "predictions" not in preds:
        return
    h, w = img.shape[:2]
    for p in preds["predictions"]:
        x = int(p["x"] - p["width"]/2)
        y = int(p["y"] - p["height"]/2)
        x_max = int(p["x"] + p["width"]/2)
        y_max = int(p["y"] + p["height"]/2)
        cls = str(p.get("class", "obj")).lower()
        conf = p.get("confidence", 0)
        color = (0,255,0) if cls in ("logo","logo_front","logo-front") else (0,0,255)
        x, y, x_max, y_max = clamp(x,0,w-1), clamp(y,0,h-1), clamp(x_max,0,w-1), clamp(y_max,0,h-1)
        cv2.rectangle(img, (x, y), (x_max, y_max), color, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (x, max(0,y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def inference_worker(client):
    global last_preds_global
    while not g_bExit:
        try:
            frame = infer_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if frame is None: break
        try:
            last_preds_global = client.infer(frame, model_id="big400-lip-insp-before-cleaning-da2vm/3")
        except Exception as e:
            print("[INFER ERROR]", e)
        infer_queue.task_done()

def work_thread(cam, client):
    global g_bExit, USE_FIT, ZOOM, last_preds_global
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 800)

    t_last_fps, frames, fcount = time.time(), 0, 0
    last_preds_global = None

    while not g_bExit:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('f'): USE_FIT = not USE_FIT
            elif key in (ord('+'), ord('=')): ZOOM = clamp(ZOOM*1.1, ZOOM_MIN, ZOOM_MAX)
            elif key in (ord('-'), ord('_')): ZOOM = clamp(ZOOM/1.1, ZOOM_MIN, ZOOM_MAX)
            continue

        try:
            img_full = frame_to_bgr(stOutFrame)
            frames += 1
            now = time.time()
            if now - t_last_fps >= 1.0:
                fps = frames / (now - t_last_fps)
                frames = 0
                t_last_fps = now

            # Resize for inference
            work_img, scale = resize_keep_aspect(img_full, INFER_MAX_W)

            # Send frame to inference thread every N frames
            fcount += 1
            if fcount % INFER_EVERY_N_FRAMES == 0:
                try:
                    infer_queue.put_nowait(work_img)
                except queue.Full:
                    pass  # skip if worker busy

            # Draw last predictions
            if last_preds_global:
                draw_predictions(work_img, last_preds_global)

            # Display with zoom/fit
            view = work_img
            if USE_FIT:
                _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
                if win_w > 0 and win_h > 0:
                    s = min(win_w / view.shape[1], win_h / view.shape[0])
                    view = cv2.resize(view, (max(1,int(view.shape[1]*s)), max(1,int(view.shape[0]*s))), interpolation=cv2.INTER_AREA)
                mode_txt = "[FIT]"
            else:
                view = cv2.resize(view, (max(1,int(view.shape[1]*ZOOM)), max(1,int(view.shape[0]*ZOOM))), interpolation=cv2.INTER_AREA)
                mode_txt = f"[ZOOM {int(ZOOM*100)}%]"

            draw_info(view, fps, mode_txt)
            cv2.imshow(WIN_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break

        finally:
            cam.MV_CC_FreeImageBuffer(stOutFrame)

    cv2.destroyAllWindows()

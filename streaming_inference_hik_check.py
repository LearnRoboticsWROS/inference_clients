# -- coding: utf-8 --

import os, sys, threading, time
from ctypes import *
import numpy as np
import cv2

# HIK SDK binding (solo MvCameraControl_class)
sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *

# Roboflow inference
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ... dopo la creazione del CLIENT
CONF = 0.3  # metti la soglia che vuoi (es. 0.60)
IOU  = 0.50  # opzionale

g_bExit = False
PIXEL_MONO8 = 0x01080001

# ----- DISPLAY STATE -----
USE_FIT = False         # False = usa zoom percentuale; True = fit alla finestra
ZOOM = 0.70             # 70% all'avvio
ZOOM_MIN, ZOOM_MAX = 0.10, 2.00
WIN_NAME = "Hikrobot + Roboflow (q: esci, f: fit/zoom, +/-: zoom)"

# ----- INFERENCE CONFIG -----
# RF_API_URL  = os.getenv("RF_API_URL",  "http://localhost:9001")
# RF_API_KEY  = os.getenv("RF_API_KEY",  "YOUR_API_KEY_HERE")       
# RF_MODEL_ID = os.getenv("RF_MODEL_ID", "glass-defect/5")          
INFER_EVERY_N_FRAMES = 2    # elabora 1 frame ogni 2
INFER_MAX_W = 1280          # ridimensiona per inferenza (mantiene aspect ratio)

#CLIENT = InferenceHTTPClient(api_url=RF_API_URL, api_key=RF_API_KEY)



CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # Assicurati che il server sia in esecuzione
    api_key="EC9puzE6crcRm7buAF1S"   # Usa la tua API key
)

CLIENT.configure(
    InferenceConfiguration(
        confidence_threshold=CONF,
        iou_threshold=IOU
    )
)

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def frame_to_bgr(stOutFrame):
    """Converte il buffer della camera in BGR numpy (assumiamo Mono8)."""
    info = stOutFrame.stFrameInfo
    w, h = info.nWidth, info.nHeight
    pt = info.enPixelType
    buf = string_at(stOutFrame.pBufAddr, info.nFrameLen)

    if pt == PIXEL_MONO8:
        img = np.frombuffer(buf, dtype=np.uint8, count=w*h).reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"PixelType non gestito: 0x{pt:x}. Imposta Mono8 o aggiungi demosaic.")

def resize_keep_aspect(img, max_w):
    """Ridimensiona img a max_w mantenendo aspect (se già più piccola, lascia)."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    scale = max_w / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

def draw_info(img, fps=None, mode_txt=""):
    if fps is not None:
        cv2.putText(img, f"{img.shape[1]}x{img.shape[0]}  FPS:{fps:.1f} {mode_txt}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)


def draw_predictions(img, preds):
    """Disegna box Roboflow su img: verde per logo/logo_front, rosso per il resto."""
    if not preds or "predictions" not in preds:
        return

    h, w = img.shape[:2]
    for p in preds["predictions"]:
        x = p.get("x"); y = p.get("y")
        ww = p.get("width"); hh = p.get("height")
        cls = p.get("class", "obj")
        conf = p.get("confidence", 0)

        if None in (x, y, ww, hh):
            continue

        x_min = int(x - ww/2); y_min = int(y - hh/2)
        x_max = int(x + ww/2); y_max = int(y + hh/2)

        # clamp ai bordi
        x_min = clamp(x_min, 0, w-1); y_min = clamp(y_min, 0, h-1)
        x_max = clamp(x_max, 0, w-1); y_max = clamp(y_max, 0, h-1)

        # --- colore: VERDE solo per logo / logo_front ---
        cls_norm = str(cls).lower().replace(" ", "")
        if cls_norm in ("logo", "logo_front", "logo-front"):
            color = (0, 255, 0)   # verde
        else:
            color = (0, 0, 255)   # rosso

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (x_min, max(0, y_min-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def work_thread(cam):
    global g_bExit, USE_FIT, ZOOM
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 800)

    t_last_fps, frames = time.time(), 0
    fps = None
    fcount = 0
    last_preds = None  # tieni l’ultimo risultato per visualizzarlo anche tra un’inferenza e l’altra

    while not g_bExit:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            # no frame → gestisci input finestra
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('f'): USE_FIT = not USE_FIT
            elif key in (ord('+'), ord('=')): ZOOM = clamp(ZOOM * 1.1, ZOOM_MIN, ZOOM_MAX)
            elif key in (ord('-'), ord('_')): ZOOM = clamp(ZOOM / 1.1, ZOOM_MIN, ZOOM_MAX)
            continue

        try:
            img_full = frame_to_bgr(stOutFrame)

            # Calcolo FPS “smoothed”
            frames += 1
            now = time.time()
            if now - t_last_fps >= 1.0:
                fps = frames / (now - t_last_fps)
                frames = 0
                t_last_fps = now

            # --- Ridimensiona per inferenza e (anche) visualizzazione di base ---
            work_img, scale = resize_keep_aspect(img_full, INFER_MAX_W)

            # ogni N frame, inferenza
            fcount += 1
            if fcount % INFER_EVERY_N_FRAMES == 0:
                try:
                    # NB: il client accetta numpy BGR
                    
                    #LIP
                    last_preds = CLIENT.infer(work_img, model_id="big400-lip-insp-before-cleaning-da2vm/3")
                    #last_preds = CLIENT.infer(work_img, model_id="flaw-detection-bs7lj/big400-lip-insp-before-cleaning-da2vm-instant-1")
                    
                    # BODY
                    # last_preds = CLIENT.infer(work_img, model_id="big400-body-insp-before-cleaning-7cr8v/2")

                    
                except Exception as e:
                    # non bloccare il ciclo se il server non risponde
                    last_preds = None

            # disegna ultimi risultati sul frame ridotto (coerente con coordinate Roboflow)
            if last_preds:
                draw_predictions(work_img, last_preds)

            # --- Visualizzazione con fit/zoom sulla versione ridotta ---
            view = work_img
            if USE_FIT:
                _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
                if win_w > 0 and win_h > 0:
                    s = min(win_w / view.shape[1], win_h / view.shape[0])
                    view = cv2.resize(view, (max(1,int(view.shape[1]*s)), max(1,int(view.shape[0]*s))),
                                      interpolation=cv2.INTER_AREA)
                mode_txt = "[FIT]"
            else:
                s = ZOOM
                view = cv2.resize(view, (max(1,int(view.shape[1]*s)), max(1,int(view.shape[0]*s))),
                                  interpolation=cv2.INTER_AREA)
                mode_txt = f"[ZOOM {int(ZOOM*100)}%]"

            draw_info(view, fps, mode_txt)
            cv2.imshow(WIN_NAME, view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('f'): USE_FIT = not USE_FIT
            elif key in (ord('+'), ord('=')): ZOOM = clamp(ZOOM * 1.1, ZOOM_MIN, ZOOM_MAX)
            elif key in (ord('-'), ord('_')): ZOOM = clamp(ZOOM / 1.1, ZOOM_MIN, ZOOM_MAX)

        finally:
            cam.MV_CC_FreeImageBuffer(stOutFrame)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # init SDK
    MvCamera.MV_CC_Initialize()
    print("SDKVersion[0x%x]" % MvCamera.MV_CC_GetSDKVersion())

    # enum
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("enum devices fail or no device! ret[0x%x]" % ret); sys.exit(1)

    print("Find %d devices!" % deviceList.nDeviceNum)
    for i in range(deviceList.nDeviceNum):
        info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == MV_USB_DEVICE:
            model = "".join(chr(b) for b in info.SpecialInfo.stUsb3VInfo.chModelName if b != 0)
            sn    = "".join(chr(b) for b in info.SpecialInfo.stUsb3VInfo.chSerialNumber if b != 0)
            print(f"\nu3v device: [{i}]  model: {model}  serial: {sn}")
        else:
            print(f"\ngige device: [{i}]")

    nConnectionNum = input("Select device NUMBER (0..%d): " % (deviceList.nDeviceNum-1))
    if not nConnectionNum.isdigit() or int(nConnectionNum) >= deviceList.nDeviceNum:
        print("input error!"); sys.exit(1)
    idx = int(nConnectionNum)

    # create & open
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
    if cam.MV_CC_CreateHandle(stDeviceList) != 0:
        print("create handle fail!"); sys.exit(1)
    if cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0) != 0:
        print("open device fail! (forse è già aperta altrove?)"); sys.exit(1)

    # Trigger OFF e formato
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetEnumValue("PixelFormat", PIXEL_MONO8)
    try:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 15.0)
    except: pass

    if cam.MV_CC_StartGrabbing() != 0:
        print("start grabbing fail!"); sys.exit(1)

    try:
        th = threading.Thread(target=work_thread, args=(cam,), daemon=True)
        th.start()
        th.join()  # premi 'q' nella finestra per terminare
    finally:
        g_bExit = True
        try: th.join(timeout=1)
        except: pass
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()
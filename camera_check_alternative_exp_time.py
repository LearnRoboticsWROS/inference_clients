# -- coding: utf-8 --

import sys, threading, time
from ctypes import *
import numpy as np
import cv2

# binding SDK (solo MvCameraControl_class)
sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *

g_bExit = False
PIXEL_MONO8 = 0x01080001

# ----- DISPLAY STATE -----
USE_FIT = False         # False = usa zoom percentuale; True = fit alla finestra
ZOOM = 0.70             # 70% all'avvio
ZOOM_MIN, ZOOM_MAX = 0.10, 2.00
WIN_NAME = "Hikrobot (q: esci, f: fit/zoom, +/-: zoom) [expo auto-toggle]"

# ----- EXPOSURE TOGGLE -----
EXPO_HIGH_US   = 500.0      # µs
EXPO_LOW_US    = 500.0       # µs
SWITCH_PERIOD  = 0.1         # secondi: 1s high, 1s low, ...

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

def frame_to_bgr(stOutFrame):
    info = stOutFrame.stFrameInfo
    w, h = info.nWidth, info.nHeight
    pt = info.enPixelType
    buf = string_at(stOutFrame.pBufAddr, info.nFrameLen)

    if pt == PIXEL_MONO8:
        img = np.frombuffer(buf, dtype=np.uint8, count=w*h).reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"PixelType non gestito: 0x{pt:x}. Imposta Mono8 o aggiungi demosaic.")

def draw_info(img, fps=None, zoom_txt="", expo_txt=""):
    y = 28
    if fps is not None:
        cv2.putText(img, f"{img.shape[1]}x{img.shape[0]}  FPS:{fps:.1f} {zoom_txt}",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        y += 30
    if expo_txt:
        cv2.putText(img, expo_txt, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def set_exposure_us(cam, us):
    # Assicurati che ExposureAuto sia OFF, poi imposta ExposureTime (µs)
    # ExposureAuto: 0=Off, 1=Once, 2=Continuous (in molte HIK)
    try:
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)
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

def work_thread(cam):
    global g_bExit, USE_FIT, ZOOM
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 800)

    # esposizione iniziale = HIGH
    current_high = True
    set_exposure_us(cam, EXPO_HIGH_US)
    last_switch = time.time()

    t0, n = time.time(), 0
    fps = None

    while not g_bExit:
        # toggle expo ogni SWITCH_PERIOD secondi
        now = time.time()
        if now - last_switch >= SWITCH_PERIOD:
            current_high = not current_high
            set_exposure_us(cam, EXPO_HIGH_US if current_high else EXPO_LOW_US)
            last_switch = now

        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            # timeout: continua a gestire input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('f'): USE_FIT = not USE_FIT
            elif key in (ord('+'), ord('=')): ZOOM = clamp(ZOOM * 1.1, ZOOM_MIN, ZOOM_MAX)
            elif key in (ord('-'), ord('_')): ZOOM = clamp(ZOOM / 1.1, ZOOM_MIN, ZOOM_MAX)
            continue

        try:
            img = frame_to_bgr(stOutFrame)
            n += 1
            if n % 10 == 0:
                fps = n / (time.time() - t0 + 1e-9)
                t0, n = time.time(), 0

            view = img
            if USE_FIT:
                _, _, win_w, win_h = cv2.getWindowImageRect(WIN_NAME)
                if win_w > 0 and win_h > 0:
                    scale = min(win_w / view.shape[1], win_h / view.shape[0])
                    new_w = max(1, int(view.shape[1] * scale))
                    new_h = max(1, int(view.shape[0] * scale))
                    view = cv2.resize(view, (new_w, new_h), interpolation=cv2.INTER_AREA)
                zoom_txt = "[FIT]"
            else:
                new_w = max(1, int(view.shape[1] * ZOOM))
                new_h = max(1, int(view.shape[0] * ZOOM))
                if new_w != view.shape[1] or new_h != view.shape[0]:
                    view = cv2.resize(view, (new_w, new_h), interpolation=cv2.INTER_AREA)
                zoom_txt = f"[ZOOM {int(ZOOM*100)}%]"

            expo_now = get_exposure_us(cam)
            expo_txt = f"EXP: {int(expo_now)} µs ({'HIGH' if current_high else 'LOW'})" if expo_now else ""

            draw_info(view, fps, zoom_txt, expo_txt)
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

    # disabilita auto exposure e imposta un valore iniziale “sicuro”
    try:
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)  # OFF
    except:
        pass
    set_exposure_us(cam, EXPO_HIGH_US)

    # facoltativo: limita FPS
    try:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 25.0)
    except:
        pass

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

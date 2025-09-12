# -- coding: utf-8 --

import sys, threading, os, termios, time
from ctypes import *
import numpy as np
import cv2

# usa il path giusto dei binding (solo MvCameraControl_class)
sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *

g_bExit = False

# GenICam PixelType (valori interi, così non servono altri moduli)
PIXEL_MONO8 = 0x01080001

def frame_to_bgr(stOutFrame):
    """Converte il buffer grezzo in BGR (numpy) — assumiamo Mono8."""
    info = stOutFrame.stFrameInfo
    w, h = info.nWidth, info.nHeight
    pt = info.enPixelType

    # copia il buffer dal puntatore C in bytes
    buf = string_at(stOutFrame.pBufAddr, info.nFrameLen)

    if pt == PIXEL_MONO8:
        img = np.frombuffer(buf, dtype=np.uint8, count=w*h).reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # fallback: mostriamo un messaggio chiaro e fermiamo
    raise RuntimeError(f"PixelType non gestito: 0x{pt:x}. "
                       f"Imposta la camera in Mono8 o aggiungi il demosaic Bayer.")

def work_thread(cam):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    t0, n = time.time(), 0

    while not g_bExit:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret == 0:
            try:
                bgr = frame_to_bgr(stOutFrame)
                n += 1
                if n % 10 == 0:
                    fps = n / (time.time() - t0 + 1e-9)
                    cv2.putText(bgr, f"{bgr.shape[1]}x{bgr.shape[0]}  FPS:{fps:.1f}",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Hikrobot Preview (q per uscire)", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            finally:
                cam.MV_CC_FreeImageBuffer(stOutFrame)
        # else: timeout, ignora

    cv2.destroyAllWindows()

def press_any_key_exit():
    fd = sys.stdin.fileno()
    old_ttyinfo = termios.tcgetattr(fd)
    new_ttyinfo = old_ttyinfo[:]
    new_ttyinfo[3] &= ~termios.ICANON
    new_ttyinfo[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    try:
        os.read(fd, 1)
    except:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)

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

    # Trigger OFF
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # FORZA PIXEL FORMAT = MONO8 (così la conversione è banale e veloce)
    ret = cam.MV_CC_SetEnumValue("PixelFormat", PIXEL_MONO8)
    if ret != 0:
        print("Warning: non sono riuscito a impostare PixelFormat=Mono8 (ret=0x%x). Continuo..." % ret)

    # (opzionale) limita FPS per non saturare
    try:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        cam.MV_CC_SetFloatValue("AcquisitionFrameRate", 15.0)
    except:
        pass

    if cam.MV_CC_StartGrabbing() != 0:
        print("start grabbing fail!"); sys.exit(1)

    try:
        th = threading.Thread(target=work_thread, args=(cam,), daemon=True)
        th.start()
        print("Anteprima attiva. Premi un tasto qui o 'q' nella finestra per uscire.")
        press_any_key_exit()
    finally:
        g_bExit = True
        th.join()
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()

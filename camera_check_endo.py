import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # 0 for endoscope

if not cap.isOpened():
    print("Error impossible to open camera.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error impossible to read the frame.")
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
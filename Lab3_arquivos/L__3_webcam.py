import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow("frame", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("x"):
        cv.imwrite("foto1.png", frame)
        print("Salvou foto1.png")

cap.release()
cv.destroyAllWindows()

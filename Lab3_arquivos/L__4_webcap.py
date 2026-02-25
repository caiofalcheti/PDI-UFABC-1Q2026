import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    raise SystemExit(1)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fps = 20.0  # escolha “adequada” (pode ajustar)
delay_ms = int(1000 / fps)

fourcc = cv.VideoWriter_fourcc(*"XVID")
out = cv.VideoWriter("saida.avi", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # REMOVIDO: frame = cv.flip(frame, 0)  # isso deixava o vídeo invertido

    out.write(frame)
    cv.imshow("frame", frame)

    key = cv.waitKey(delay_ms) & 0xFF
    if key == ord("q"):
        break

cap.release()
out.release()
cv.destroyAllWindows()

import time
import cv2 as cv

VIDEO_PATH = "big_buck_bunny.mp4"
speed = 1.0  # 2.0 = mais rápido; 0.5 = mais lento

cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Erro abrindo o vídeo:", VIDEO_PATH)
    raise SystemExit(1)

fps = cap.get(cv.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 25.0  # fallback

base_dt = 1.0 / fps
dt = base_dt / speed  # maior dt = mais lento

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow("frame", frame)

    time.sleep(dt)  # controla velocidade

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

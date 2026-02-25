import cv2 as cv

img_color = cv.imread("messi5.jpg", cv.IMREAD_COLOR)
if img_color is None:
    print("Erro ao abrir messi5.jpg")
    raise SystemExit(1)

cv.imshow("image", img_color)
k = cv.waitKey(0) & 0xFF

if k == 27:  # ESC
    cv.destroyAllWindows()
elif k == ord("s"):
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    cv.imwrite("messigray.png", img_gray)
    cv.destroyAllWindows()

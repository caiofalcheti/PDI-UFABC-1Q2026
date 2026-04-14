import cv2
import numpy as np
import pytesseract
import shutil

pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

def processar_imagem(img_original):
    img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    img_equalizada = cv2.equalizeHist(img_blur)
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(img_equalizada, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def detectar_placa(img_original, img_blackhat):
    threshold, img_bin = cv2.threshold(img_blackhat, 127, 255, cv2.THRESH_BINARY)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    img_fechada = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_aberta = cv2.morphologyEx(img_fechada, cv2.MORPH_OPEN, kernel_open)
    contornos, _ = cv2.findContours(img_aberta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        proporcao = w / float(h)
        area = w * h
        if 2.0 < proporcao < 6.0 and 2000 < area < 60000:
            candidatos.append((x, y, w, h))

    if candidatos:
        x, y, w, h = max(candidatos, key=lambda b: b[2]*b[3])
        pad_w = int(0.20 * w)
        pad_h = int(0.35 * h)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_original.shape[1], x + w + pad_w)
        y1 = min(img_original.shape[0], y + h + pad_h)
        placa = img_original[y0:y1, x0:x1]
        if placa is not None and placa.size > 0:
            return placa
    return None

def isolar_caracteres(placa):
    placa_cinza = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    placa_blur = cv2.GaussianBlur(placa_cinza, (3,3), 0)
    placa_equalizada = cv2.equalizeHist(placa_blur)
    _, placa_bin = cv2.threshold(placa_equalizada, 127, 255, cv2.THRESH_BINARY)
    placa_bin_inv = cv2.bitwise_not(placa_bin)
    contornos, _ = cv2.findContours(placa_bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = placa_bin_inv.shape
    boxes = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if (0.40 * H <= h <= 0.95 * H) and (0.03 * W <= w <= 0.25 * W) and (0.15 <= (w / float(h)) <= 0.9):
            boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[0])

    chars = []
    for (x, y, w, h) in boxes:
        ch = placa_bin_inv[y:y+h, x:x+w]
        ch = cv2.copyMakeBorder(ch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        ch = cv2.medianBlur(ch, 3)
        ch = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        ch = 255 - ch  # inverte para texto preto em fundo branco
        ch = cv2.copyMakeBorder(ch, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
        ch = cv2.resize(ch, (50, 80), interpolation=cv2.INTER_NEAREST)
        chars.append((x, y, w, h, ch))

    return chars, placa_bin_inv, boxes

def ocr_tesseract(chars):
    lidos = []
    config = r'--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    for (_, _, _, _, ch) in chars:
        img_rgb = cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB)
        txt = pytesseract.image_to_string(img_rgb, config=config)
        txt = ''.join([c for c in txt.upper() if c.isalnum()])[:1]
        lidos.append(txt if txt else '')
    return ''.join(lidos)

def exibir_caracteres(chars):
    """Cria uma imagem com todos os caracteres isolados lado a lado."""
    if not chars:
        return None
    altura = 80
    largura_total = sum(ch.shape[1] for _, _, _, _, ch in chars) + 10 * (len(chars) - 1)
    painel = 255 * np.ones((altura, largura_total), dtype=np.uint8)
    x_atual = 0
    for _, _, _, _, ch in chars:
        h, w = ch.shape
        painel[0:h, x_atual:x_atual+w] = ch
        x_atual += w + 10
    return painel

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        exit()

    print("Pressione 'g' para capturar e processar a placa.")
    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura do frame.")
            break

        cv2.imshow("Webcam - pressione 'g' para capturar", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('g'):
            print("Processando frame...")
            img_original = frame.copy()
            img_blackhat = processar_imagem(img_original)
            placa = detectar_placa(img_original, img_blackhat)

            if placa is None:
                print("Nenhuma placa detectada.")
                continue

            chars, placa_bin_inv, boxes = isolar_caracteres(placa)

            # Desenha retângulos na placa (apenas para visualização)
            placa_vis = placa.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(placa_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

            # Exibe as imagens com OpenCV
            cv2.imshow("Placa (ROI)", placa)
            cv2.imshow("Placa binaria invertida", placa_bin_inv)
            cv2.imshow("Placa com caixas", placa_vis)

            painel_chars = exibir_caracteres(chars)
            if painel_chars is not None:
                cv2.imshow("Caracteres isolados", painel_chars)
            else:
                print("Nenhum caractere isolado.")

            texto = ocr_tesseract(chars)
            print(f"Placa OCR: {texto}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np
import pytesseract
import shutil

# =============================================================================
# CONFIGURAÇÕES GERAIS (ALTERE AQUI)
# =============================================================================
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

# Parâmetros de detecção da placa
PROPORCAO_MIN = 1.5
PROPORCAO_MAX = 7.0
AREA_MIN = 1000
AREA_MAX = 100000
PAD_W_FATOR = 0.20
PAD_H_FATOR = 0.35

# Kernel para fechamento (horizontal x vertical)
KERNEL_CLOSE = (50, 20)
ITERACOES_CLOSE = 2

# Parâmetros para isolamento de caracteres (duas linhas)
ALTURA_MIN_FATOR = 0.25   # Em relação à altura da placa
ALTURA_MAX_FATOR = 0.60
LARGURA_MIN_FATOR = 0.02
LARGURA_MAX_FATOR = 0.25
PROPORCAO_CHAR_MIN = 0.1
PROPORCAO_CHAR_MAX = 1.2

# Configuração Tesseract (PSM 7 para uma linha de texto)
TESSERACT_CONFIG = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO
# =============================================================================
def processar_imagem(img_original):
    img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    img_equalizada = cv2.equalizeHist(img_blur)
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(img_equalizada, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def detectar_placa(img_original, img_blackhat):
    # Binarização com Otsu + inversão
    _, img_bin = cv2.threshold(img_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)

    # Fechamento com kernel grande
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_CLOSE)
    img_fechada = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_close, iterations=ITERACOES_CLOSE)

    # Dilatação leve para fechar buracos
    kernel_dilate = np.ones((3,3), np.uint8)
    img_fechada = cv2.dilate(img_fechada, kernel_dilate, iterations=1)

    # Abertura para remover ruídos
    kernel_open = np.ones((3,3), np.uint8)
    img_aberta = cv2.morphologyEx(img_fechada, cv2.MORPH_OPEN, kernel_open)

    contornos, _ = cv2.findContours(img_aberta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        proporcao = w / float(h)
        area = w * h
        if PROPORCAO_MIN < proporcao < PROPORCAO_MAX and AREA_MIN < area < AREA_MAX:
            candidatos.append((x, y, w, h, area))

    if candidatos:
        x, y, w, h, _ = max(candidatos, key=lambda b: b[4])
        pad_w = int(PAD_W_FATOR * w)
        pad_h = int(PAD_H_FATOR * h)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_original.shape[1], x + w + pad_w)
        y1 = min(img_original.shape[0], y + h + pad_h)
        placa = img_original[y0:y1, x0:x1]
        return placa, (x0, y0, x1, y1), img_bin, img_fechada, img_aberta
    return None, None, None, None, None

def isolar_caracteres_duas_linhas(placa):
    """Separa caracteres em duas linhas e retorna listas de imagens prontas para OCR."""
    placa_cinza = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    placa_blur = cv2.GaussianBlur(placa_cinza, (3,3), 0)
    placa_equalizada = cv2.equalizeHist(placa_blur)
    _, placa_bin = cv2.threshold(placa_equalizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placa_bin_inv = cv2.bitwise_not(placa_bin)

    # Remove ruídos pequenos
    kernel = np.ones((2,2), np.uint8)
    placa_bin_inv = cv2.morphologyEx(placa_bin_inv, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(placa_bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = placa_bin_inv.shape
    boxes = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if (ALTURA_MIN_FATOR * H <= h <= ALTURA_MAX_FATOR * H and
            LARGURA_MIN_FATOR * W <= w <= LARGURA_MAX_FATOR * W and
            PROPORCAO_CHAR_MIN <= (w / float(h)) <= PROPORCAO_CHAR_MAX):
            boxes.append((x, y, w, h))

    if not boxes:
        return [], [], placa_bin_inv, []

    # Separa em duas linhas pela mediana de y
    media_y = np.median([b[1] for b in boxes])
    linha_sup = [b for b in boxes if b[1] < media_y]
    linha_inf = [b for b in boxes if b[1] >= media_y]

    linha_sup = sorted(linha_sup, key=lambda b: b[0])
    linha_inf = sorted(linha_inf, key=lambda b: b[0])

    def extrair_caracteres(linha):
        chars = []
        for (x, y, w, h) in linha:
            ch = placa_bin_inv[y:y+h, x:x+w]
            ch = cv2.copyMakeBorder(ch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            ch = cv2.resize(ch, (40, 60), interpolation=cv2.INTER_NEAREST)
            ch = 255 - ch  # fundo branco
            chars.append(ch)
        return chars

    chars_sup = extrair_caracteres(linha_sup)
    chars_inf = extrair_caracteres(linha_inf)

    return chars_sup, chars_inf, placa_bin_inv, boxes

def ocr_linha(chars_linha):
    if not chars_linha:
        return ""
    # Concatena caracteres horizontalmente
    altura = 60
    espaco = 5
    largura_total = sum(ch.shape[1] for ch in chars_linha) + espaco * (len(chars_linha) - 1)
    img_linha = 255 * np.ones((altura, largura_total), dtype=np.uint8)
    x_atual = 0
    for ch in chars_linha:
        h, w = ch.shape
        img_linha[0:h, x_atual:x_atual+w] = ch
        x_atual += w + espaco

    texto = pytesseract.image_to_string(img_linha, config=TESSERACT_CONFIG)
    texto = ''.join([c for c in texto.upper() if c.isalnum()])
    return texto

def exibir_painel_caracteres(chars_sup, chars_inf):
    """Cria uma imagem com todos os caracteres das duas linhas lado a lado."""
    todos = chars_sup + chars_inf
    if not todos:
        return None
    altura = 60
    largura_total = sum(ch.shape[1] for ch in todos) + 10 * (len(todos) - 1)
    painel = 255 * np.ones((altura, largura_total), dtype=np.uint8)
    x_atual = 0
    for ch in todos:
        h, w = ch.shape
        painel[0:h, x_atual:x_atual+w] = ch
        x_atual += w + 10
    return painel

# =============================================================================
# PROGRAMA PRINCIPAL (WEBCAM)
# =============================================================================
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
            break

        cv2.imshow("Webcam - pressione 'g'", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('g'):
            print("Processando frame...")
            img_original = frame.copy()
            img_blackhat = processar_imagem(img_original)
            placa, retangulo, img_bin, img_fechada, img_aberta = detectar_placa(img_original, img_blackhat)

            if placa is None:
                print("Nenhuma placa detectada.")
                continue

            # Mostra imagens intermediárias (OpenCV)
            cv2.imshow("Binarizada + Inversa", img_bin)
            cv2.imshow("Fechamento", img_fechada)
            cv2.imshow("Abertura", img_aberta)

            chars_sup, chars_inf, placa_bin_inv, boxes = isolar_caracteres_duas_linhas(placa)

            # Desenha caixas na placa
            placa_vis = placa.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(placa_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

            cv2.imshow("Placa (ROI)", placa)
            cv2.imshow("Placa binaria invertida", placa_bin_inv)
            cv2.imshow("Placa com caixas", placa_vis)

            painel = exibir_painel_caracteres(chars_sup, chars_inf)
            if painel is not None:
                cv2.imshow("Caracteres isolados", painel)
            else:
                print("Nenhum caractere isolado.")

            texto_sup = ocr_linha(chars_sup)
            texto_inf = ocr_linha(chars_inf)
            texto_final = f"{texto_sup} {texto_inf}".strip()
            print(f"Placa OCR: {texto_final}")

            # Desenha retângulo da placa no frame original
            if retangulo:
                x0, y0, x1, y1 = retangulo
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
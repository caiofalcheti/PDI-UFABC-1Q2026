import cv2
import numpy as np
import pytesseract
import shutil
import time

# =============================================================================
# CONFIGURAÇÕES GERAIS (ALTERE AQUI)
# =============================================================================
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

# Intervalo entre capturas (em segundos)
INTERVALO_PROCESSAMENTO = 2.5   # ← AUMENTE ESTE VALOR PARA REDUZIR A FREQUÊNCIA

# Largura de redimensionamento (melhora desempenho)
LARGURA_PROCESSAMENTO = 640

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
# FUNÇÕES DE PROCESSAMENTO (OTIMIZADAS PARA DUAS LINHAS)
# =============================================================================
def processar_imagem(img_original):
    img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    img_equalizada = cv2.equalizeHist(img_blur)
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(img_equalizada, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def detectar_placa(img_original, img_blackhat):
    _, img_bin = cv2.threshold(img_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_CLOSE)
    img_fechada = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_close, iterations=ITERACOES_CLOSE)
    kernel_dilate = np.ones((3,3), np.uint8)
    img_fechada = cv2.dilate(img_fechada, kernel_dilate, iterations=1)
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
        return placa
    return None

def isolar_caracteres_duas_linhas(placa):
    placa_cinza = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    placa_blur = cv2.GaussianBlur(placa_cinza, (3,3), 0)
    placa_equalizada = cv2.equalizeHist(placa_blur)
    _, placa_bin = cv2.threshold(placa_equalizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placa_bin_inv = cv2.bitwise_not(placa_bin)

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
        return [], []

    media_y = np.median([b[1] for b in boxes])
    linha_sup = sorted([b for b in boxes if b[1] < media_y], key=lambda b: b[0])
    linha_inf = sorted([b for b in boxes if b[1] >= media_y], key=lambda b: b[0])

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
    return chars_sup, chars_inf

def ocr_linha(chars_linha):
    if not chars_linha:
        return ""
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
    return ''.join([c for c in texto.upper() if c.isalnum()])

def processar_frame_ocr(frame):
    """Executa todo o pipeline e retorna o texto da placa (ou string vazia)."""
    altura, largura = frame.shape[:2]
    if largura > LARGURA_PROCESSAMENTO:
        proporcao = LARGURA_PROCESSAMENTO / largura
        nova_altura = int(altura * proporcao)
        frame_red = cv2.resize(frame, (LARGURA_PROCESSAMENTO, nova_altura))
    else:
        frame_red = frame.copy()

    img_blackhat = processar_imagem(frame_red)
    placa = detectar_placa(frame_red, img_blackhat)

    if placa is None:
        return ""

    chars_sup, chars_inf = isolar_caracteres_duas_linhas(placa)
    texto_sup = ocr_linha(chars_sup)
    texto_inf = ocr_linha(chars_inf)
    return f"{texto_sup} {texto_inf}".strip()

# =============================================================================
# PROGRAMA PRINCIPAL (WEBCAM COM INTERVALO AUTOMÁTICO)
# =============================================================================
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        exit()

    print(f"Leitura automática a cada {INTERVALO_PROCESSAMENTO} segundos.")
    print("Pressione 'q' para sair.")

    ultimo_tempo = 0
    texto_placa = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura do frame.")
            break

        agora = time.time()

        # Processa apenas no intervalo definido
        if agora - ultimo_tempo >= INTERVALO_PROCESSAMENTO:
            texto_placa = processar_frame_ocr(frame.copy())
            ultimo_tempo = agora
            if texto_placa:
                print(f"Placa lida: {texto_placa}")

        # Exibe o texto da placa sobreposto no vídeo
        if texto_placa:
            cv2.putText(frame, f"Placa: {texto_placa}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Webcam - Leitura automatica de placa", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
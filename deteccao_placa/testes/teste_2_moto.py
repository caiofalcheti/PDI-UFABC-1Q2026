import cv2
import numpy as np
import pytesseract
import time

# =============================================================================
# CONFIGURAÇÕES GERAIS (AJUSTE CONFORME NECESSÁRIO)
# =============================================================================
# Descomente e ajuste o caminho do Tesseract se necessário (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

INTERVALO_PROCESSAMENTO = 2.5       # segundos entre cada tentativa de leitura
LARGURA_PROCESSAMENTO = 640         # redimensionamento para melhor desempenho

# Parâmetros para detecção da placa
PROPORCAO_MIN, PROPORCAO_MAX = 1.5, 7.0
AREA_MIN, AREA_MAX = 1000, 100000
PAD_W_FATOR, PAD_H_FATOR = 0.20, 0.35
KERNEL_CLOSE = (50, 20)
ITERACOES_CLOSE = 2

# Parâmetros para isolamento dos caracteres (duas linhas)
ALTURA_MIN_FATOR, ALTURA_MAX_FATOR = 0.15, 0.60
LARGURA_MIN_FATOR, LARGURA_MAX_FATOR = 0.02, 0.30
PROPORCAO_CHAR_MIN, PROPORCAO_CHAR_MAX = 0.2, 1.5
AREA_MIN_CHAR = 50

# Configuração do Tesseract (PSM 8: palavra única)
TESSERACT_CONFIG = r'--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO
# =============================================================================
def processar_imagem(img):
    """Pré-processamento para realçar a região da placa."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    equal = cv2.equalizeHist(blur)
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(equal, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def detectar_placa(img_original, img_blackhat):
    """Localiza a placa na imagem e retorna a região de interesse (ROI)."""
    bin_img = cv2.adaptiveThreshold(img_blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_CLOSE)
    fechada = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_close, iterations=ITERACOES_CLOSE)
    aberta = cv2.morphologyEx(fechada, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    contornos, _ = cv2.findContours(aberta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        proporcao = w / h
        area = w * h
        if PROPORCAO_MIN < proporcao < PROPORCAO_MAX and AREA_MIN < area < AREA_MAX:
            candidatos.append((x, y, w, h, area))

    if candidatos:
        # Seleciona o maior candidato por área
        x, y, w, h, _ = max(candidatos, key=lambda b: b[4])
        pad_w, pad_h = int(PAD_W_FATOR * w), int(PAD_H_FATOR * h)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_original.shape[1], x + w + pad_w)
        y1 = min(img_original.shape[0], y + h + pad_h)
        return img_original[y0:y1, x0:x1]
    return None

def refinar_caractere(roi_bin):
    """
    Refina um caractere individual:
    - Remove ruídos
    - Redimensiona para tamanho fixo
    - Garante texto preto em fundo branco
    - Adiciona borda branca
    """
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    roi = cv2.resize(roi, (50, 80), interpolation=cv2.INTER_CUBIC)
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    roi = 255 - roi  # inverte para texto preto em fundo branco
    roi = cv2.copyMakeBorder(roi, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return roi

def isolar_caracteres_duas_linhas(placa):
    """
    Segmenta os caracteres da placa e os separa em duas linhas
    (superior e inferior).
    """
    if placa is None or placa.size == 0:
        return [], []

    gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_placa = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    bin_placa = cv2.morphologyEx(bin_placa, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)

    H, W = bin_placa.shape
    contornos, _ = cv2.findContours(bin_placa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        # Filtros baseados nas proporções da placa
        if not (ALTURA_MIN_FATOR * H <= h <= ALTURA_MAX_FATOR * H):
            continue
        if not (LARGURA_MIN_FATOR * W <= w <= LARGURA_MAX_FATOR * W):
            continue
        if not (PROPORCAO_CHAR_MIN <= w/h <= PROPORCAO_CHAR_MAX):
            continue
        if w * h < AREA_MIN_CHAR:
            continue
        boxes.append((x, y, w, h))

    if len(boxes) < 3:   # mínimo esperado de caracteres
        return [], []

    # Separação das duas linhas pela mediana da coordenada Y central
    centros_y = [b[1] + b[3]/2 for b in boxes]
    mediana_y = np.median(centros_y)
    linha_sup = [b for b in boxes if (b[1] + b[3]/2) < mediana_y]
    linha_inf = [b for b in boxes if (b[1] + b[3]/2) >= mediana_y]

    # Ordena cada linha da esquerda para a direita
    linha_sup.sort(key=lambda b: b[0])
    linha_inf.sort(key=lambda b: b[0])

    def extrair(lista):
        chars = []
        for (x, y, w, h) in lista:
            roi = bin_placa[y:y+h, x:x+w]
            chars.append(refinar_caractere(roi))
        return chars

    return extrair(linha_sup), extrair(linha_inf)

def ocr_linha(chars):
    """Realiza OCR em uma linha de caracteres montada em uma única imagem."""
    if not chars:
        return ""
    altura = 100
    espaco = 12
    largura = sum(c.shape[1] for c in chars) + espaco * (len(chars) - 1)
    img = 255 * np.ones((altura, largura), dtype=np.uint8)
    x = 0
    for ch in chars:
        h, w = ch.shape
        y_off = (altura - h) // 2
        img[y_off:y_off+h, x:x+w] = ch
        x += w + espaco

    texto = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
    return ''.join(c for c in texto.upper() if c.isalnum())

def processar_frame_ocr(frame):
    """Pipeline completo: pré-processamento → detecção da placa → OCR."""
    altura, largura = frame.shape[:2]
    if largura > LARGURA_PROCESSAMENTO:
        proporcao = LARGURA_PROCESSAMENTO / largura
        frame = cv2.resize(frame, (LARGURA_PROCESSAMENTO, int(altura * proporcao)))

    img_blackhat = processar_imagem(frame)
    placa = detectar_placa(frame, img_blackhat)

    if placa is None:
        return ""

    sup, inf = isolar_caracteres_duas_linhas(placa)
    texto_sup = ocr_linha(sup)
    texto_inf = ocr_linha(inf)
    resultado = f"{texto_sup} {texto_inf}".strip()
    return resultado

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        exit()

    print(f"Leitura automática a cada {INTERVALO_PROCESSAMENTO} segundos.")
    print("Pressione 'q' na janela da webcam para sair.")

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

        # Exibe o vídeo sem sobreposição de texto
        cv2.imshow("Webcam - Leitura automatica de placa (moto)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
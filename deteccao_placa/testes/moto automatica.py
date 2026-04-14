import cv2
import numpy as np
import pytesseract
import shutil
import time

# =============================================================================
# CONFIGURAÇÕES AJUSTADAS PARA PLACAS DE MOTO (E CARRO)
# =============================================================================
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

INTERVALO_PROCESSAMENTO = 2.0
LARGURA_PROCESSAMENTO = 640

# Parâmetros de detecção de placa (agora aceita proporções mais quadradas)
PROPORCAO_MIN = 0.8   # Antes era 2.0 (apenas horizontais)
PROPORCAO_MAX = 5.0   # Mantém compatibilidade com carros
AREA_MIN = 1500       # Reduzido para placas de moto menores
AREA_MAX = 80000
PAD_W_FATOR = 0.15
PAD_H_FATOR = 0.20    # Menor expansão vertical para não capturar muito fundo

# Kernel de fechamento menor para evitar unir as duas linhas
KERNEL_CLOSE_W = 15   # Antes 25
KERNEL_CLOSE_H = 3    # Antes 5

# Parâmetros para caracteres (agora consideram duas linhas)
ALTURA_MIN_FATOR = 0.25   # Caracteres podem ser menores em altura relativa
ALTURA_MAX_FATOR = 0.60   # Pois há duas linhas, cada uma ocupa ~metade da altura
LARGURA_MIN_FATOR = 0.02
LARGURA_MAX_FATOR = 0.20
PROPORCAO_CHAR_MIN = 0.10
PROPORCAO_CHAR_MAX = 1.2

# Configuração Tesseract para linha de caracteres (PSM 7 trata uma linha de texto)
TESSERACT_CONFIG = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# =============================================================================
# FUNÇÕES MODIFICADAS
# =============================================================================
def processar_imagem(img_original):
    img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    img_equalizada = cv2.equalizeHist(img_blur)
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(img_equalizada, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def detectar_placa(img_original, img_blackhat):
    _, img_bin = cv2.threshold(img_blackhat, 127, 255, cv2.THRESH_BINARY)
    # Kernel de fechamento ajustável (menor para não juntar linhas)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_CLOSE_W, KERNEL_CLOSE_H))
    img_fechada = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_aberta = cv2.morphologyEx(img_fechada, cv2.MORPH_OPEN, kernel_open)
    contornos, _ = cv2.findContours(img_aberta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        proporcao = w / float(h)
        area = w * h
        if (PROPORCAO_MIN < proporcao < PROPORCAO_MAX and AREA_MIN < area < AREA_MAX):
            candidatos.append((x, y, w, h, area))

    if candidatos:
        # Seleciona o maior candidato (por área)
        x, y, w, h, _ = max(candidatos, key=lambda b: b[4])
        pad_w = int(PAD_W_FATOR * w)
        pad_h = int(PAD_H_FATOR * h)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_original.shape[1], x + w + pad_w)
        y1 = min(img_original.shape[0], y + h + pad_h)
        placa = img_original[y0:y1, x0:x1]
        if placa is not None and placa.size > 0:
            return placa
    return None

def isolar_caracteres_duas_linhas(placa):
    """Segmenta caracteres e os separa em duas linhas (superior e inferior)."""
    placa_cinza = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    placa_blur = cv2.GaussianBlur(placa_cinza, (3,3), 0)
    placa_equalizada = cv2.equalizeHist(placa_blur)
    _, placa_bin = cv2.threshold(placa_equalizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    placa_bin_inv = cv2.bitwise_not(placa_bin)

    # Remove pequenos ruídos
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

    # Separa em duas linhas baseado na coordenada y
    media_y = np.mean([b[1] for b in boxes])
    linha_sup = [b for b in boxes if b[1] < media_y]
    linha_inf = [b for b in boxes if b[1] >= media_y]

    # Ordena cada linha da esquerda para a direita
    linha_sup = sorted(linha_sup, key=lambda b: b[0])
    linha_inf = sorted(linha_inf, key=lambda b: b[0])

    # Extrai e processa cada caractere
    chars_sup = []
    for (x, y, w, h) in linha_sup:
        ch = placa_bin_inv[y:y+h, x:x+w]
        ch = cv2.copyMakeBorder(ch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        ch = cv2.resize(ch, (40, 60), interpolation=cv2.INTER_NEAREST)
        ch = 255 - ch  # fundo branco
        chars_sup.append(ch)

    chars_inf = []
    for (x, y, w, h) in linha_inf:
        ch = placa_bin_inv[y:y+h, x:x+w]
        ch = cv2.copyMakeBorder(ch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        ch = cv2.resize(ch, (40, 60), interpolation=cv2.INTER_NEAREST)
        ch = 255 - ch
        chars_inf.append(ch)

    return chars_sup, chars_inf

def ocr_linha(chars_linha):
    """Aplica OCR em uma lista de imagens de caracteres (já em fundo branco)."""
    if not chars_linha:
        return ""
    # Concatena os caracteres horizontalmente para formar uma palavra
    altura = 60
    espaco = 5
    largura_total = sum(ch.shape[1] for ch in chars_linha) + espaco * (len(chars_linha) - 1)
    img_linha = 255 * np.ones((altura, largura_total), dtype=np.uint8)
    x_atual = 0
    for ch in chars_linha:
        h, w = ch.shape
        img_linha[0:h, x_atual:x_atual+w] = ch
        x_atual += w + espaco

    config = TESSERACT_CONFIG
    texto = pytesseract.image_to_string(img_linha, config=config)
    texto = ''.join([c for c in texto.upper() if c.isalnum()])
    return texto

def processar_frame_ocr(frame):
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

    # Junta as duas linhas (pode ser exibido como "ABC1234" ou "ABC-1234")
    if texto_sup and texto_inf:
        return f"{texto_sup} {texto_inf}"
    elif texto_sup:
        return texto_sup
    elif texto_inf:
        return texto_inf
    return ""

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        exit()

    print(f"Leitura automática a cada {INTERVALO_PROCESSAMENTO}s. Pressione 'q' para sair.")
    ultimo_tempo = 0
    texto_placa = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        agora = time.time()
        if agora - ultimo_tempo >= INTERVALO_PROCESSAMENTO:
            texto_placa = processar_frame_ocr(frame.copy())
            ultimo_tempo = agora
            if texto_placa:
                print(f"Placa lida: {texto_placa}")

        if texto_placa:
            cv2.putText(frame, f"Placa: {texto_placa}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Webcam - Placa Moto/Carro", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
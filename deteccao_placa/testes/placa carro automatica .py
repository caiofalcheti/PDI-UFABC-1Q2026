import cv2
import numpy as np
import pytesseract
import shutil
import time

# =============================================================================
# CONFIGURAÇÕES GERAIS (ALTERE AQUI PARA AJUSTAR O FUNCIONAMENTO)
# =============================================================================
# Caminho do executável do Tesseract (caso não esteja no PATH)
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

# Intervalo entre processamentos OCR (em segundos)
INTERVALO_PROCESSAMENTO = 2.0

# Largura para redimensionamento do frame antes do processamento (melhora desempenho)
LARGURA_PROCESSAMENTO = 640

# Parâmetros do pré‑processamento da imagem
KERNEL_BLACKHAT = (15, 15)      # Tamanho do kernel para BlackHat (destaca regiões)
GAUSSIAN_BLUR = (5, 5)          # Tamanho do kernel do desfoque Gaussiano

# Parâmetros para detecção da região da placa
PROPORCAO_MIN = 2.0             # Proporção mínima largura/altura para candidato a placa
PROPORCAO_MAX = 6.0             # Proporção máxima largura/altura
AREA_MIN = 2000                 # Área mínima (em pixels) do contorno candidato
AREA_MAX = 60000                # Área máxima do contorno candidato
PAD_W_FATOR = 0.20              # Fator de expansão horizontal da ROI (20% para cada lado)
PAD_H_FATOR = 0.35              # Fator de expansão vertical da ROI (35% para cada lado)

# Parâmetros para isolamento dos caracteres dentro da placa
ALTURA_MIN_FATOR = 0.40         # Altura mínima do caractere em relação à altura da placa
ALTURA_MAX_FATOR = 0.95         # Altura máxima do caractere
LARGURA_MIN_FATOR = 0.03        # Largura mínima do caractere em relação à largura da placa
LARGURA_MAX_FATOR = 0.25        # Largura máxima do caractere
PROPORCAO_CHAR_MIN = 0.15       # Proporção mínima largura/altura do caractere
PROPORCAO_CHAR_MAX = 0.9        # Proporção máxima largura/altura do caractere

# Configuração do Tesseract para reconhecimento de caracteres individuais
TESSERACT_CONFIG = r'--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO (NÃO PRECISA ALTERAR)
# =============================================================================
def processar_imagem(img_original):
    """Aplica pré‑processamento para realçar possíveis regiões de placa."""
    img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_cinza, GAUSSIAN_BLUR, 0)
    img_equalizada = cv2.equalizeHist(img_blur)
    kernel = np.ones(KERNEL_BLACKHAT, np.uint8)
    blackhat = cv2.morphologyEx(img_equalizada, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def detectar_placa(img_original, img_blackhat):
    """Localiza a região da placa usando contornos e filtros geométricos."""
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
        if (PROPORCAO_MIN < proporcao < PROPORCAO_MAX and
            AREA_MIN < area < AREA_MAX):
            candidatos.append((x, y, w, h))

    if candidatos:
        x, y, w, h = max(candidatos, key=lambda b: b[2]*b[3])
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

def isolar_caracteres(placa):
    """Segmenta os caracteres dentro da placa e os prepara para OCR."""
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
        altura_rel = h / float(H)
        largura_rel = w / float(W)
        proporcao = w / float(h)
        if (ALTURA_MIN_FATOR <= altura_rel <= ALTURA_MAX_FATOR and
            LARGURA_MIN_FATOR <= largura_rel <= LARGURA_MAX_FATOR and
            PROPORCAO_CHAR_MIN <= proporcao <= PROPORCAO_CHAR_MAX):
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
    return chars

def ocr_tesseract(chars):
    """Executa OCR Tesseract em cada caractere isolado e retorna a string."""
    lidos = []
    for (_, _, _, _, ch) in chars:
        img_rgb = cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB)
        txt = pytesseract.image_to_string(img_rgb, config=TESSERACT_CONFIG)
        txt = ''.join([c for c in txt.upper() if c.isalnum()])[:1]
        lidos.append(txt if txt else '')
    return ''.join(lidos)

def processar_frame_ocr(frame):
    """Executa o pipeline completo de OCR em uma cópia do frame."""
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

    chars = isolar_caracteres(placa)
    if not chars:
        return ""

    return ocr_tesseract(chars)

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        exit()

    print("Pressione 'q' para sair.")
    print(f"Placa será lida automaticamente a cada {INTERVALO_PROCESSAMENTO} segundos.")

    ultimo_tempo = 0
    texto_placa = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha na captura do frame.")
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

        cv2.imshow("Webcam - Leitura automatica de placa", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np
import easyocr
import sys
import os

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
# Nome do arquivo de imagem (pode ser alterado aqui ou passado como argumento)
NOME_ARQUIVO = "placa.jpg"

LARGURA_PROCESSAMENTO = 640

# Parâmetros de detecção da placa
PROPORCAO_MIN, PROPORCAO_MAX = 1.5, 7.0
AREA_MIN, AREA_MAX = 1000, 100000
PAD_W_FATOR, PAD_H_FATOR = 0.20, 0.35
KERNEL_CLOSE = (50, 20)
ITERACOES_CLOSE = 2

# Parâmetros para isolamento de caracteres (duas linhas)
ALTURA_MIN_FATOR, ALTURA_MAX_FATOR = 0.15, 0.60
LARGURA_MIN_FATOR, LARGURA_MAX_FATOR = 0.02, 0.30
PROPORCAO_CHAR_MIN, PROPORCAO_CHAR_MAX = 0.2, 1.5
AREA_MIN_CHAR = 50

# Inicializa o EasyOCR (apenas inglês, sem GPU)
print("Inicializando EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR pronto!\n")

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO DE IMAGEM
# =============================================================================
def processar_imagem(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    equal = cv2.equalizeHist(blur)
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(equal, cv2.MORPH_BLACKHAT, kernel)
    return blackhat

def detectar_placa(img_original, img_blackhat):
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
        x, y, w, h, _ = max(candidatos, key=lambda b: b[4])
        pad_w, pad_h = int(PAD_W_FATOR * w), int(PAD_H_FATOR * h)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_original.shape[1], x + w + pad_w)
        y1 = min(img_original.shape[0], y + h + pad_h)
        return img_original[y0:y1, x0:x1]
    return None

def refinar_caractere(roi_bin):
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    roi = cv2.resize(roi, (50, 80), interpolation=cv2.INTER_CUBIC)
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    roi = 255 - roi
    roi = cv2.copyMakeBorder(roi, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    return roi

def isolar_caracteres_duas_linhas(placa):
    if placa is None or placa.size == 0:
        return [], [], None, [], None

    gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_placa = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    bin_placa = cv2.morphologyEx(bin_placa, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    placa_bin_inv = bin_placa.copy()

    H, W = bin_placa.shape
    contornos, _ = cv2.findContours(bin_placa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if not (ALTURA_MIN_FATOR * H <= h <= ALTURA_MAX_FATOR * H):
            continue
        if not (LARGURA_MIN_FATOR * W <= w <= LARGURA_MAX_FATOR * W):
            continue
        if not (PROPORCAO_CHAR_MIN <= w/h <= PROPORCAO_CHAR_MAX):
            continue
        if w * h < AREA_MIN_CHAR:
            continue
        boxes.append((x, y, w, h))

    if len(boxes) < 3:
        return [], [], placa_bin_inv, boxes, None

    centros_y = [b[1] + b[3]/2 for b in boxes]
    mediana_y = np.median(centros_y)
    linha_sup = [b for b in boxes if (b[1] + b[3]/2) < mediana_y]
    linha_inf = [b for b in boxes if (b[1] + b[3]/2) >= mediana_y]

    linha_sup.sort(key=lambda b: b[0])
    linha_inf.sort(key=lambda b: b[0])

    placa_vis = placa.copy()
    for (x, y, w, h) in linha_sup:
        cv2.rectangle(placa_vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
    for (x, y, w, h) in linha_inf:
        cv2.rectangle(placa_vis, (x, y), (x+w, y+h), (255, 0, 0), 1)

    def extrair(lista):
        chars = []
        for (x, y, w, h) in lista:
            roi = bin_placa[y:y+h, x:x+w]
            chars.append(refinar_caractere(roi))
        return chars

    chars_sup = extrair(linha_sup)
    chars_inf = extrair(linha_inf)
    return chars_sup, chars_inf, placa_bin_inv, boxes, placa_vis

def exibir_caracteres_isolados(chars_sup, chars_inf):
    todos_chars = chars_sup + chars_inf
    if not todos_chars:
        return None
    altura = 80
    largura_total = sum(ch.shape[1] for ch in todos_chars) + 10 * (len(todos_chars) - 1)
    painel = 255 * np.ones((altura, largura_total), dtype=np.uint8)
    x_atual = 0
    for ch in todos_chars:
        h, w = ch.shape
        painel[0:h, x_atual:x_atual+w] = ch
        x_atual += w + 10
    return painel

def ocr_linha_easyocr(chars):
    if not chars:
        return ""
    altura = 100
    espaco = 12
    largura = sum(c.shape[1] for c in chars) + espaco * (len(chars) - 1)
    img_linha = 255 * np.ones((altura, largura), dtype=np.uint8)
    x = 0
    for ch in chars:
        h, w = ch.shape
        y_off = (altura - h) // 2
        img_linha[y_off:y_off+h, x:x+w] = ch
        x += w + espaco

    resultados = reader.readtext(img_linha, detail=0, paragraph=True)
    texto = resultados[0] if resultados else ""
    return ''.join(c for c in texto.upper() if c.isalnum())

def ocr_placa_inteira(placa):
    if placa is None:
        return ""
    gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, bin_placa = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bin_placa) > 127:
        bin_placa = 255 - bin_placa
    resultados = reader.readtext(bin_placa, detail=0, paragraph=True)
    texto = " ".join(resultados) if resultados else ""
    return ''.join(c for c in texto.upper() if c.isalnum())

def processar_imagem_arquivo(caminho_imagem):
    """Processa uma imagem do disco e exibe resultados."""
    if not os.path.exists(caminho_imagem):
        print(f"Erro: Arquivo '{caminho_imagem}' não encontrado.")
        return

    frame = cv2.imread(caminho_imagem)
    if frame is None:
        print(f"Erro: Não foi possível ler a imagem '{caminho_imagem}'.")
        return

    print(f"Processando imagem: {caminho_imagem}")
    texto, placa, bin_img, placa_vis, painel = processar_frame_completo(frame)

    if texto:
        print(f"Placa OCR: {texto}")
    else:
        print("Nenhuma placa detectada ou texto vazio.")

    # Exibe a imagem original e as janelas de depuração
    cv2.imshow("Imagem Original", frame)
    if placa is not None:
        cv2.imshow("Placa (ROI)", placa)
    if bin_img is not None:
        cv2.imshow("Placa binaria invertida", bin_img)
    if placa_vis is not None:
        cv2.imshow("Placa com caixas (verde=sup, azul=inf)", placa_vis)
    if painel is not None:
        cv2.imshow("Caracteres isolados", painel)

    print("\nPressione qualquer tecla nas janelas para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def processar_frame_completo(frame):
    """Pipeline completo, retorna texto e imagens de depuração."""
    altura, largura = frame.shape[:2]
    if largura > LARGURA_PROCESSAMENTO:
        proporcao = LARGURA_PROCESSAMENTO / largura
        frame_red = cv2.resize(frame, (LARGURA_PROCESSAMENTO, int(altura * proporcao)))
    else:
        frame_red = frame.copy()

    img_blackhat = processar_imagem(frame_red)
    placa = detectar_placa(frame_red, img_blackhat)

    if placa is None:
        return "", None, None, None, None

    chars_sup, chars_inf, placa_bin_inv, boxes, placa_vis = isolar_caracteres_duas_linhas(placa)
    painel = exibir_caracteres_isolados(chars_sup, chars_inf)

    texto_sup = ocr_linha_easyocr(chars_sup)
    texto_inf = ocr_linha_easyocr(chars_inf)
    resultado_duas_linhas = f"{texto_sup} {texto_inf}".strip()

    if len(resultado_duas_linhas.replace(" ", "")) < 4:
        resultado_direto = ocr_placa_inteira(placa)
        if len(resultado_direto) >= 4:
            return resultado_direto, placa, placa_bin_inv, placa_vis, painel
        else:
            return resultado_duas_linhas, placa, placa_bin_inv, placa_vis, painel
    else:
        return resultado_duas_linhas, placa, placa_bin_inv, placa_vis, painel

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    # Permite passar o nome do arquivo como argumento
    if len(sys.argv) > 1:
        arquivo = sys.argv[1]
    else:
        arquivo = NOME_ARQUIVO

    processar_imagem_arquivo(arquivo)
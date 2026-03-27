#Edson Felipe RA 201922149
# Caio Falcheti RA 11201920936
#Nicolas da Costa Vidal	RA 11201811472

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

import shutil
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

def processar_imagem(img_original):
    
    img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Reduzir Ruídos
    img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 0)
    img_equalizada = cv2.equalizeHist(img_blur)

    # Filtro para destacar regiões
    kernel = np.ones((15, 15), np.uint8)
    blackhat = cv2.morphologyEx(img_equalizada, cv2.MORPH_BLACKHAT, kernel)

    return blackhat


def detectar_placa(img_original, img_blackhat):

    # Binarização com Otsu
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    _, img_bin = cv2.threshold(img_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)

    # Detecção de bordas com Canny
    bordas = cv2.Canny(img_bin, 70, 150)

    # Fechamento "em cruz" (horizontal + vertical de uma vez)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 20))
    img_fechada = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel_cross, iterations=2)

    # Dilatação leve para fechar buracos
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_fechada = cv2.dilate(img_fechada, kernel_dilate, iterations=1)

    # Abertura para remover pequenos ruídos
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_aberta = cv2.morphologyEx(img_fechada, cv2.MORPH_OPEN, kernel_open)

    # Encontrar contornos externos
    contornos, _ = cv2.findContours(img_aberta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Procurar candidatos a placa
    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        proporcao = w / float(h)
        area = w * h
        if 1.5 < proporcao < 7.0 and 1000 < area < 100000:
            candidatos.append((x, y, w, h))

    placa = None
    if candidatos:
        x, y, w, h = max(candidatos, key=lambda b: b[2] * b[3])
        pad_w = int(0.20 * w)
        pad_h = int(0.35 * h)

        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_original.shape[1], x + w + pad_w)
        y1 = min(img_original.shape[0], y + h + pad_h)

        placa = img_original[y0:y1, x0:x1]

        # Desenhar retângulo na imagem original
        cv2.rectangle(img_rgb, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # ---- Etapas para visualização ----
    etapas = [
        ("Original", img_rgb, 'rgb'),
        ("Binarizada", img_bin, 'gray'),
        ("Bordas (Canny)", bordas, 'gray'),
        ("Fechamento", img_fechada, 'gray'),
        ("Abertura", img_aberta, 'gray')
    ]

    plt.figure(figsize=(15, 4))
    for i, (titulo, img, tipo) in enumerate(etapas):
        plt.subplot(1, len(etapas), i + 1)
        if tipo == 'gray':
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(titulo)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    # ----------------------------------

    return placa
    
def isolar_caracteres(placa):

    placa_cinza = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

    # Reduzir Ruídos
    placa_blur = cv2.GaussianBlur(placa_cinza, (3,3), 0)
    placa_equalizada = cv2.equalizeHist(placa_blur)

    # Binarização da Placa
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
        ch = placa_bin_inv[y:y+h, x:x+w]        # binária (caractere branco)

        # limpeza mínima p/ OCR
        ch = cv2.copyMakeBorder(ch, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        ch = cv2.medianBlur(ch, 3)
        ch = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
        # ch = cv2.erode(ch, np.ones((2,2), np.uint8), 1)

        # >>> inverter para EasyOCR: texto preto em fundo branco
        ch = 255 - ch

        # normalização p/ OCR
        ch = cv2.copyMakeBorder(ch, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)  # agora fundo é branco
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

if __name__ == '__main__':
    # pasta_imagens = r"img\Carro"
    pasta_imagens = r"img\Moto"
    lista_imagens = os.listdir(pasta_imagens)

    for img in lista_imagens:
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_caminho = os.path.join(pasta_imagens, img)
            img_original = cv2.imread(img_caminho)

            img_blackhat = processar_imagem(img_original)
            placa = detectar_placa(img_original, img_blackhat)

            if placa is None or placa.size == 0:
                print(f"[{img}] -> Nenhuma placa detectada (ROI vazio).")
                continue

            chars, placa_bin_inv, boxes = isolar_caracteres(placa)

            placa_vis = placa.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(placa_vis, (x, y), (x+w, y+h), (0,255,0), 1)

            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(placa, cv2.COLOR_BGR2RGB)); plt.title('Placa (ROI)'); plt.axis('off')
            plt.subplot(1,3,2); plt.imshow(placa_bin_inv, cmap='gray'); plt.title('Placa binária (inv)'); plt.axis('off')
            plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(placa_vis, cv2.COLOR_BGR2RGB)); plt.title('Chars (caixas)'); plt.axis('off')
            plt.show()

            if chars:
                plt.figure(figsize=(14,3))
                for i, (_,_,_,_,ch) in enumerate(chars):
                    plt.subplot(1, len(chars), i+1)
                    plt.imshow(ch, cmap='gray'); plt.axis('off')
                plt.suptitle('Caracteres isolados')
                plt.show()
            else:
                print('Nenhum caractere isolado.')
                continue

            texto = ocr_tesseract(chars)
            print(f"[{img}] -> Placa OCR: {texto}")

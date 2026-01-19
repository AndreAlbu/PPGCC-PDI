from collections import deque
import numpy as np
import cv2

def crescimento_regiao_cinza(img, x_seed, y_seed, limiar, conectividade):
 
  h, w = img.shape
  intensidade_seed = int(img[y_seed, x_seed])

  visitado = np.zeros((h, w), dtype=np.uint8)
  mascara = np.zeros((h, w), dtype=np.uint8)

  fila = deque()
  fila.append((y_seed, x_seed))
  visitado[y_seed, x_seed] = 1

  if conectividade == 8:

    vizinhos = [(-1,-1), (-1,0), (-1,1),
                (0,-1),          (0,1),
                (1,-1),  (1,0),  (1,1)]
  else:

    vizinhos = [(-1,0), (1,0), (0,-1), (0,1)]

  while fila:

    y, x = fila.popleft()

    if abs(int(img[y, x]) - intensidade_seed) <= limiar:

      mascara[y, x] = 255

      for dy, dx in vizinhos:

        ny, nx = y + dy, x + dx

        if 0 <= ny < h and 0 <= nx < w and visitado[ny, nx] == 0:

          visitado[ny, nx] = 1

          fila.append((ny, nx))

  return mascara, intensidade_seed

def main_29(img, limiar, conectividade):

  janela = "Q29"

  img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  overlay = img_color.copy()
  mascara_atual = np.zeros_like(img)

  def atualiza_overlay():

    nonlocal overlay

    overlay = img_color.copy()

    overlay[mascara_atual == 255] = (0, 255, 0)

  def on_mouse(event, x, y, flags, param):

    nonlocal mascara_atual

    if event == cv2.EVENT_LBUTTONDOWN:

      mascara_atual, intensidade_seed = crescimento_regiao_cinza(img, x, y, limiar, conectividade)
      atualiza_overlay()

      print(f"Seed=({x},{y}) | Intensidade seed={intensidade_seed} | "f"Pixels na regiao={int(np.sum(mascara_atual==255))}")

  cv2.namedWindow(janela)
  cv2.setMouseCallback(janela, on_mouse)

  atualiza_overlay()

  while True:

    cv2.imshow(janela, overlay)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('q'):

      break

    if tecla == ord('r'):

      mascara_atual[:] = 0

      atualiza_overlay()

  cv2.destroyAllWindows()

  return mascara_atual

img = cv2.imread("house.tif", cv2.IMREAD_GRAYSCALE)

mascara_q29 = main_29(img, 15, 10)

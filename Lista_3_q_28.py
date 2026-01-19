import cv2
import numpy as np
from collections import deque

def crescimento_regiao_binaria(img_binaria, x_seed, y_seed, conectividade):
  
  h, w = img_binaria.shape
  valor_seed = int(img_binaria[y_seed, x_seed])

  visitado = np.zeros((h, w), dtype=np.uint8)
  mascara = np.zeros((h, w), dtype=np.uint8)

  fila = deque()
  fila.append((y_seed, x_seed))
  visitado[y_seed, x_seed] = 1

  if conectividade == 8:

    viz = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

  else:

    viz = [(-1,0), (1,0), (0,-1), (0,1)]

  while fila:

    y, x = fila.popleft()

    if int(img_binaria[y, x]) == valor_seed:

      mascara[y, x] = 255

      for dy, dx in viz:

        ny, nx = y + dy, x + dx

        if 0 <= ny < h and 0 <= nx < w and visitado[ny, nx] == 0:

          visitado[ny, nx] = 1
          fila.append((ny, nx))

  return mascara, valor_seed

def mostra_crescimento_regiao_por_clique(img_binaria, conectividade):
  
  janela = "Q28"

  img_vis = cv2.cvtColor(img_binaria, cv2.COLOR_GRAY2BGR)
  overlay = img_vis.copy()
  mascara_atual = np.zeros_like(img_binaria)

  def atualiza_overlay():

    nonlocal overlay

    overlay = img_vis.copy()

    #pinta a regiao em verde
    overlay[mascara_atual == 255] = (0, 255, 0)

  def on_mouse(event, x, y, flags, param):

    nonlocal mascara_atual

    if event == cv2.EVENT_LBUTTONDOWN:

      mascara_atual, valor_seed = crescimento_regiao_binaria(img_binaria, x, y, conectividade=conectividade)
      atualiza_overlay()
      print(f"Seed=({x},{y}) | valor_seed={valor_seed} | pixels_regiao={int(np.sum(mascara_atual==255))}")

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

img = cv2.imread("dog-3.gif", cv2.IMREAD_GRAYSCALE)

mascara_regiao = mostra_crescimento_regiao_por_clique(img, 8)

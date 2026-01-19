import cv2
import numpy as np

def rastreia_caderno_hsv(indice_camera=0):

  cap = cv2.VideoCapture(indice_camera)
  if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam.")

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.flip(frame, 1)

    # Conversão para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Faixa de cor ROSA (ajustável conforme iluminação)
    lower_rosa = np.array([140, 50, 50], dtype=np.uint8)
    upper_rosa = np.array([175, 255, 255], dtype=np.uint8)

    mascara = cv2.inRange(hsv, lower_rosa, upper_rosa)

    # Pós-processamento para reduzir ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    # Encontra contornos
    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contornos:
      maior_contorno = max(contornos, key=cv2.contourArea)

      if cv2.contourArea(maior_contorno) > 1500:

        momentos = cv2.moments(maior_contorno)

        if momentos["m00"] != 0:
          cx = int(momentos["m10"] / momentos["m00"])
          cy = int(momentos["m01"] / momentos["m00"])

          # Desenha centróide
          cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
          cv2.putText(
              frame,
              f"Centroide: ({cx}, {cy})",
              (cx + 10, cy),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.6,
              (0, 255, 0),
              2
          )

          # Desenha contorno
          cv2.drawContours(frame, [maior_contorno], -1, (0, 255, 0), 2)

    # Visualização
    cv2.imshow("Frame", frame)
    cv2.imshow("Mascara HSV (Rosa)", mascara)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


# Executar localmente:
rastreia_caderno_hsv(0)

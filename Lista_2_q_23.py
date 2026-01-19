import cv2
import numpy as np

def mascara_pele_hsv(frame_bgr):

  hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

  #Ajuste
  lower = np.array([0, 40, 60], dtype=np.uint8)
  upper = np.array([25, 255, 255], dtype=np.uint8)

  mascara = cv2.inRange(hsv, lower, upper)
  
  return mascara

def mascara_pele_ycrcb(frame_bgr):

  ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

  #Ajuste
  lower = np.array([0, 133, 77], dtype=np.uint8)
  upper = np.array([255, 173, 127], dtype=np.uint8)
  mascara = cv2.inRange(ycrcb, lower, upper)
  
  return mascara

def pos_processa_mascara(mascara):

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  mascara = cv2.medianBlur(mascara, 5)
  mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
  mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

  return mascara

def aplica_mascara(frame_bgr, mascara):
  
  return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mascara)

def detecta_pele_webcam(indice_camera):

  cap = cv2.VideoCapture(indice_camera)

  if not cap.isOpened():

    raise RuntimeError("Erro camera")

  while True:

    ok, frame = cap.read()

    if not ok:

      break

    frame = cv2.flip(frame, 1)

    m_hsv = pos_processa_mascara(mascara_pele_hsv(frame))
    m_ycc = pos_processa_mascara(mascara_pele_ycrcb(frame))

    res_hsv = aplica_mascara(frame, m_hsv)
    res_ycc = aplica_mascara(frame, m_ycc)

    linha1 = np.hstack([frame, res_hsv, res_ycc])

    m_hsv_3 = cv2.cvtColor(m_hsv, cv2.COLOR_GRAY2BGR)
    m_ycc_3 = cv2.cvtColor(m_ycc, cv2.COLOR_GRAY2BGR)
    linha2 = np.hstack([np.zeros_like(frame), m_hsv_3, m_ycc_3])

    saida = np.vstack([linha1, linha2])

    cv2.putText(saida, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(saida, "HSV", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(saida, "YCrCb", (2*frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(saida, "Mascara HSV", (frame.shape[1] + 10, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(saida, "Mascara YCrCb", (2*frame.shape[1] + 10, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("q_23", saida)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

detecta_pele_webcam(0)

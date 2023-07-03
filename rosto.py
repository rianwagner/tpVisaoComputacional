# Trabalho da diciplina de visão computacional - UFSJ
# Desenvolvido usando os recursos do mediapipe
import cv2 
import mediapipe as mp 

# Ativa a webcam
webcam = cv2.VideoCapture(0) 

# ativando a solução de reconhecimento de rosto
reconhecimento_rosto = mp.solutions.face_detection 
# Solução de desenho de rosto
desenho = mp.solutions.drawing_utils 
# criando o item que consegue ler uma imagem e reconhecer os rostos
reconhecedor_rosto = reconhecimento_rosto.FaceDetection() 

while webcam.isOpened():
    # lê a imagem da webcam
    validacao, frame = webcam.read()
    if not validacao:
        break
    imagem = frame
    # usa o reconhecedor para criar uma lista com os rostos reconhecidos
    lista_rostos = reconhecedor_rosto.process(imagem)
    # caso algum rosto tenha sido reconhecido
    if lista_rostos.detections: 
        for rosto in lista_rostos.detections: 
            # desenha o rosto na imagem
            desenho.draw_detection(imagem, rosto)
    
    cv2.imshow("Rostos na sua webcam", imagem) 
    if cv2.waitKey(5) == 27: # ESC para encerrar
        break
# encerra a conexão com a webcam
webcam.release() 
# fecha a janela que mostra o que a webcam está vendo
cv2.destroyAllWindows() 
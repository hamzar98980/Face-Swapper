import numpy as np
import os
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Inicialización de la aplicación
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640,640))

# Modelo de intercambio de rostros
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

# Función para intercambiar rostros en un video
def swap_faces_in_video(video_path, img_path, output_dir, app, swapper):
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Leer el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    # Obtener propiedades del video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurar el escritor de video para el video de salida
    output_video_path = os.path.join(output_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el archivo de salida
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Leer la imagen de origen y detectar el rostro
    img_source = cv2.imread(img_path)
    face_source = app.get(img_source)[0]

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar el rostro en el frame actual
        faces = app.get(frame)
        if len(faces) > 0:
            face_target = faces[0]  # Supone que hay solo un rostro en cada frame
            # Intercambiar rostros
            frame_swapped = swapper.get(frame, face_target, face_source, paste_back=True)
        else:
            frame_swapped = frame

        # Escribir el frame intercambiado en el video de salida
        out.write(frame_swapped)

        frame_count += 1
        print(f"Procesando frame {frame_count}/{total_frames}")

    # Liberar recursos
    cap.release()
    out.release()
    print("Intercambio de rostros completado. Video guardado en:", output_video_path)

# Ejemplo de uso
output_directory = 'ruta/destino/'
video_path = 'ruta/demo_video.mp4'
image_path = 'ruta/swap_image.jpg'
swap_faces_in_video(video_path, image_path, output_directory, app, swapper)

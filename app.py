# Importaciones necesarias
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Inicialización de la aplicación
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640,640))

# Modelo de intercambio de rostros
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False,download_zip=False)

# Función para intercambiar rostros y mostrar imágenes
def swap_n_show(img1_fn, img2_fn, app, swapper,output_dir, plot_before=True, plot_after=True):

  # Lectura de imágenes
  img1 = cv2.imread(img1_fn)
  img2 = cv2.imread(img2_fn)

  # Mostrar imágenes antes del intercambio (opcional)
  if plot_before:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img1[:, :, ::-1])
    axs[0].axis('off')
    axs[1].imshow(img2[:, :, ::-1])
    axs[1].axis('off')
    plt.show()

  # Detección de rostros
  face1 = app.get(img1)[0]
  face2 = app.get(img2)[0]

  # Copias de las imágenes para evitar modificaciones en las originales
  img1_ = img1.copy()
  img2_ = img2.copy()

  # Intercambio de rostros
  img1 = swapper.get(img1_, face1, face2, paste_back=True)
  img2 = swapper.get(img2_, face2, face1, paste_back=True)

  # Guardar las imágenes intercambiadas en el directorio especificado
  cv2.imwrite(os.path.join(output_dir, 'img1_swapped.jpg'), img1 )
  cv2.imwrite(os.path.join(output_dir, 'img2_swapped.jpg'), img2)
    
  # Mostrar imágenes después del intercambio (opcional)
  if plot_after:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img1[:, :, ::-1])  # Usar img1 en lugar de img1_
    axs[0].axis('off')
    axs[1].imshow(img2[:, :, ::-1])  # Usar img2 en lugar de img2_
    axs[1].axis('off')
    plt.show()

  # Retorno de las imágenes intercambiadas
  return img1_, img2_

# Ejemplo de uso
output_directory = 'ruta/destino/'
os.makedirs(output_directory, exist_ok=True)

# Ejemplo de uso
_ = swap_n_show('ruta/imagen1.jpg', 'ruta/imagen2.jpg', app, swapper,output_directory )

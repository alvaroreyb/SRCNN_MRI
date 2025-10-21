import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from PIL import Image
import random
directory = "path\coronal"#/sagittal /axial
train_dir = r"train"
testLR_dir = r"TestLR"
testHR_dir = r"TestHR"

for dir in [train_dir, testLR_dir, testHR_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".npy")]

data = [np.load(file_path) for file_path in file_paths]
train_data, temp_test_data = train_test_split(data, test_size=0.3, random_state=42)
testHR_data = temp_test_data

def procesar_y_guardar(datos, directorio_destino):
    for i, mri_data in enumerate(datos):
        j = random.randint(7,19)
        imagen_pil = Image.fromarray(mri_data[j, :, :])
        nombre_archivo = f"{i}c.png"
        ruta_guardado = os.path.join(directoryDest, nombre_archivo)
        imagen_pil.save(ruta_guardado)
"""
procesar_y_guardar(train_data, train_dir)
procesar_y_guardar(testHR_data, testLR_dir)
procesar_y_guardar(testHR_data, testHR_dir)
"""

directory = r"valid\coronal" #coronal/axial for the other folders
directoryDest = r"\valid"
file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".npy")]
data = [np.load(file_path) for file_path in file_paths]

for i, mri_data in enumerate(data):
    j = random.randint(7,19)
    imagen_pil = Image.fromarray(mri_data[j, :, :])
    nombre_archivo = f"{i}c.png"
    ruta_guardado = os.path.join(directoryDest, nombre_archivo)
    imagen_pil.save(ruta_guardado)



import os
import numpy as np
from PIL import Image
from tqdm import tqdm  

"""
This function takes an image as input and divides it into smaller patches of a specified size (tamano_patch) 
with a certain overlap (solapamiento) between them. It returns these patches as a NumPy array.
"""
def crear_patches(imagen, tamano_patch=32, solapamiento=0.5):
    alto, ancho = imagen.shape
    paso_vertical = int(tamano_patch * solapamiento)
    paso_horizontal = int(tamano_patch * solapamiento)
    patches = []
    for i in range(0, alto - tamano_patch + 1, paso_vertical):
        for j in range(0, ancho - tamano_patch + 1, paso_horizontal):
            patch = imagen[i:i+tamano_patch, j:j+tamano_patch]
            patches.append(patch)
    return np.array(patches)
"""
 This function processes all the images in a given input folder (carpeta_entrada), 
 generates patches for each image using the crear_patches function, 
 and saves these patches as separate images in a specified output folder (carpeta_salida). 
 It also includes a progress bar to monitor the process and displays the count of patches created.
"""

def procesar_carpeta(carpeta_entrada, carpeta_salida, tamano_patch=32, solapamiento=0.5):
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    archivos = [f for f in os.listdir(carpeta_entrada) if os.path.isfile(os.path.join(carpeta_entrada, f))]
    contador_patches = 0
    pbar = tqdm(archivos, desc="Patches imgs")
    for archivo in pbar:
        ruta_completa = os.path.join(carpeta_entrada, archivo)
        imagen = Image.open(ruta_completa)
        imagen_array = np.array(imagen)
        patches = crear_patches(imagen_array, tamano_patch, solapamiento)
        for i, patch in enumerate(patches):
            patch_imagen = Image.fromarray(patch)
            patch_imagen.save(os.path.join(carpeta_salida, f"{archivo}_{i}.png"))
            contador_patches += 1
            pbar.set_postfix(parches_creados=contador_patches)

carpeta_entrada = r"" 
carpeta_salida = r""

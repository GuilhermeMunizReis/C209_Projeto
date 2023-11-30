from PIL import Image
import numpy as np
from antialiasing import kernel_gaussiano, filtro_gaussiano, grayscale

img = Image.open("serrilhada.png")
img_np = np.array(img)

img_gray = grayscale(img_np)

kernel_size = 5
sigma = 50.0
mascara_gaussiana = kernel_gaussiano(kernel_size, sigma)

img_filtrada = filtro_gaussiano(img_gray, mascara_gaussiana)

img_inicial = Image.fromarray(img_gray)
img_final = Image.fromarray(img_filtrada)


img_inicial.save("serrilhada_cinza.png")
img_final.save("processada.png")



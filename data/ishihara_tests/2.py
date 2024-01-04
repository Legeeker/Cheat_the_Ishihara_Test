import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Charger l'image
frame2 = cv2.imread("2.jpg")

# Convertir l'image BGR en HSV
hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

# Définir les plages de couleurs pour le vert en HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
mask_green = cv2.inRange(hsv2, lower_green, upper_green)
green_masked_frame2 = cv2.bitwise_and(frame2, frame2, mask=mask_green)

# Appliquer le masque vert à l'image
masked_frame2 = cv2.bitwise_and(frame2, frame2, mask=mask_green)

# Convertir l'image en niveaux de gris
gray_frame2 = cv2.cvtColor(masked_frame2, cv2.COLOR_BGR2GRAY)

# Appliquer un seuillage
_, thresh2 = cv2.threshold(gray_frame2, 127, 255, cv2.THRESH_BINARY)

# Trouver les contours dans l'image seuillée
contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dessiner les contours sur l'image masquée
contour_masked_image2 = masked_frame2.copy()
cv2.drawContours(contour_masked_image2, contours2, -1, (0, 255, 0), 2)

# Appliquer une opération de fermeture pour retirer les bruits
kernel = np.ones((5, 5), np.uint8)
contour_masked_image2_closed = cv2.morphologyEx(contour_masked_image2, cv2.MORPH_CLOSE, kernel)

# Reconnaissance optique de caractères (OCR) avec Tesseract
custom_config2 = r'--oem 3 --psm 6 outputbase digits'
detected_text2 = pytesseract.image_to_string(contour_masked_image2_closed, config=custom_config2)

# Utiliser une expression régulière pour extraire les chiffres
numbers = re.findall(r'\b\d+\b', detected_text2)

# Filtrer les chiffres pour obtenir uniquement le chiffre voulu
filtered_numbers = [num for num in numbers if num == '2']

# Afficher les résultats dans une première fenêtre
images2 = [frame2, green_masked_frame2, masked_frame2, gray_frame2, thresh2, contour_masked_image2, contour_masked_image2_closed]
titles2 = [
    'Input Image 2', 'Green Masked Image', 'Masked Image 2', 'Grayscale Image 2',
    'Thresholded Image 2', 'Contour Detection on Masked Image 2', 'Closed Contour Image 2'
]

plt.figure(figsize=(18, 12))
for i in range(7):
    plt.subplot(2, 4, i + 1)
    plt.imshow(cv2.cvtColor(images2[i], cv2.COLOR_BGR2RGB) if i in [0, 1, 2, 5, 6] else images2[i], cmap='gray')
    plt.title(titles2[i])
    plt.axis('off')

plt.show()

# Afficher le chiffre extrait
if filtered_numbers:
    print(f'Detected Number: {filtered_numbers[0]}')
else:
    print('Aucun chiffre détecté.')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Charger l'image
image_path = "data/ishihara_plates/12.jpg"

# Lire l'image
frame2 = cv2.imread(image_path)

# Convertir l'image BGR en HSV
hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

# Définir les plages de couleurs pour le jaune, le marron et le rouge
lower_yellow2 = np.array([20, 100, 100])
upper_yellow2 = np.array([30, 255, 255])

lower_brown2 = np.array([5, 50, 50])
upper_brown2 = np.array([20, 150, 150])

lower_red2 = np.array([0, 100, 100])
upper_red2 = np.array([5, 255, 255])

# Créer les masques pour chaque plage de couleur
mask_yellow2 = cv2.inRange(hsv2, lower_yellow2, upper_yellow2)
mask_brown2 = cv2.inRange(hsv2, lower_brown2, upper_brown2)
mask_red2 = cv2.inRange(hsv2, lower_red2, upper_red2)

# Combiner les masques pour obtenir le masque final
final_mask2 = cv2.bitwise_or(cv2.bitwise_or(mask_yellow2, mask_brown2), mask_red2)

# Appliquer le masque final à l'image
masked_frame2 = cv2.bitwise_and(frame2, frame2, mask=final_mask2)

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

# Afficher les résultats dans une première fenêtre
images2 = [frame2, masked_frame2, gray_frame2, thresh2, contour_masked_image2, contour_masked_image2_closed]
titles2 = [
    'Input Image 2', 'Masked Image 2', 'Grayscale Image 2',
    'Thresholded Image 2', 'Contour Detection on Masked Image 2', 'Closed Contour Image 2'
]

plt.figure(figsize=(18, 12))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images2[i], cv2.COLOR_BGR2RGB) if i in [0, 1, 4, 5] else images2[i], cmap='gray')
    plt.title(titles2[i])
    plt.axis('off')

plt.show()

# Afficher le résultat du nombre détecté pour la deuxième image
print(f'Detected Number: {detected_text2}')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Function to process the image
def process_frame(frame):
    # Convert the BGR image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for orange
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([20, 255, 255])

    # Create a mask for the orange color
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply the orange mask to the image
    orange_masked_frame = cv2.bitwise_and(frame, frame, mask=mask_orange)

    # Convert the masked image to grayscale
    gray_frame = cv2.cvtColor(orange_masked_frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the masked image
    contour_masked_image = orange_masked_frame.copy()
    cv2.drawContours(contour_masked_image, contours, -1, (0, 255, 0), 2)

    # Apply a closing operation to remove noise
    kernel = np.ones((5, 5), np.uint8)
    contour_masked_image_closed = cv2.morphologyEx(contour_masked_image, cv2.MORPH_CLOSE, kernel)

    # Perform OCR on the closed contour image
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    detected_text = pytesseract.image_to_string(contour_masked_image_closed, config=custom_config)

    return orange_masked_frame, contour_masked_image_closed, detected_text

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Process the frame
    orange_masked_frame, result_image, detected_text = process_frame(frame)

    # Superimpose the result on the actual image
    cv2.putText(frame, f'Detected Number: {detected_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Orange Masked Frame', orange_masked_frame)
    cv2.imshow('Result Image', result_image)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

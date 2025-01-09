import pytesseract as pyt
import cv2
import numpy as np

# Read the image from which text needs to be extracted
img = cv2.imread("words.jpg")

if img is None:
    print("Error: Image file not found.")
    exit()

# Specify the path to Tesseract-OCR
pyt.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Step 1: Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Use median filtering to handle noise and preserve curves
median_filtered = cv2.medianBlur(gray, 3)

# Step 3: Apply adaptive thresholding for better segmentation
thresh = cv2.adaptiveThreshold(
    median_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
)

# Step 4: Invert the image to make text white on a black background
inverted = cv2.bitwise_not(thresh)

# Step 5: Apply morphological operations for text enhancement
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))  # Small kernel for curvy and tight text
cleaned = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=2)

# Step 6: Upscale the image for better OCR accuracy
scale_factor = 3  # Larger scale for challenging text
resized = cv2.resize(cleaned, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

# Step 7: Apply sharpening to enhance text boundaries
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening filter
sharpened = cv2.filter2D(resized, -1, sharpen_kernel)

# Step 8: Extract text using Tesseract with appropriate configuration
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\'" '  # Restrict to known characters
text = pyt.image_to_string(sharpened, config=custom_config)

print("Extracted Text:")
print(text)

# Optional: Save the processed image for debugging
cv2.imwrite("processed_image_complex_text.jpg", sharpened)

# Save the extracted text to a file
with open("extracted_text_complex_fonts.txt", "w", encoding="utf-8") as f:
    f.write(text)

import cv2
import pytesseract
import pyttsx3

# Configure the path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Function to process the frame and extract text
def recognize_and_speak(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing (blurring and thresholding)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Recognize text using Tesseract
    text = pytesseract.image_to_string(thresh, config="--psm 6")
    
    # Speak the recognized text
    if text.strip():
        print("Recognized Text:", text)
        engine.say(text)
        engine.runAndWait()
    else:
        print("No text recognized.")

# Start video capture from the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read from camera.")
        break

    # Show the live video feed
    cv2.imshow("Camera Feed - Text Recognition", frame)

    # Recognize text and speak it
    recognize_and_speak(frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

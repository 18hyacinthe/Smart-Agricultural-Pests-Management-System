import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load your pre-trained classification model
model = load_model('pest_classifier_model.h5')

# Define your classes
classes = ['ants', 'bees', 'beetle', 'caterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']

# Function to preprocess the image for EfficientNetB0 
def preprocess_image(img):
    if img is None:
        return None

    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    
    img = np.expand_dims(img, axis=0)
    return img

# Function to send an email
def send_email(predicted_class, confidence):
    from_address = "# Your email address"
    to_address = "# Recipient's email address"
    subject = "Pest Detected!"
    body = f"A pest has been detected.\n\nType: {predicted_class}\nConfidence: {confidence:.2f}"

    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_address, '# Enter here your password')  # Replace with your password or app password
        text = msg.as_string()
        server.sendmail(from_address, to_address, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to perform classification and overlay the results on the frame
def classify_frame(frame):
    processed_frame = preprocess_image(frame)

    if processed_frame is None:
        return frame

    predictions = model.predict(processed_frame)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    label_text = f"Class: {predicted_class} (Confidence: {confidence:.2f})"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Send an email if the confidence is above a certain threshold
    if confidence > 0.8:  # Confidence threshold, adjust according to your needs
        send_email(predicted_class, confidence)

    return frame

# Open the webcam (index 2 for Iriun)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    classified_frame = classify_frame(frame)

    cv2.imshow('Webcam Classification', classified_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

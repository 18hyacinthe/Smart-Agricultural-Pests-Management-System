import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load your pre-trained classification model
model = load_model('pest_classication.h5')

# Define your classes
classes = ['ants', 'bees', 'beetle', 'caterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']

# Function to preprocess the image for EfficientNetB0
def preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    return img

# Function to perform classification and overlay results on the frame
def classify_frame(frame):
    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Make prediction
    predictions = model.predict(processed_frame)

    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions[0])

    # Get the class label
    predicted_class = classes[predicted_class_index]

    # Get the confidence score
    confidence = predictions[0][predicted_class_index]

    # Overlay predictions on the frame
    label_text = f"Class: {predicted_class} (Confidence: {confidence:.2f})"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# Open the video file (replace 'your_video_file.mp4' with the path to your video file)
cap = cv2.VideoCapture('vid.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video is finished
    if not ret:
        break

    # Classify the frame
    classified_frame = classify_frame(frame)

    # Display the frame
    cv2.imshow('Video Classification', classified_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()

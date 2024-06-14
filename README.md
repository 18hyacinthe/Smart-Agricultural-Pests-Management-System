# Smart Agricultural Pests Management System

## Introduction
Welcome to the Smart Agricultural Pests Management System! This project aims to revolutionize how farmers monitor and manage pests in their crops using advanced AI models and IoT technologies. By integrating a pre-trained EfficientNetV2 model with real-time video analysis and smart sensors, we provide a comprehensive solution to keep crops healthy and pest-free.

## Key Features
- **Real-time Pest Detection:** Utilize the state-of-the-art EfficientNetV2 model to identify pests in real-time.
- **Live Video Monitoring:** Seamlessly integrate with a webcam to provide continuous monitoring of crops.
- **Smart Notifications:** Receive instant email alerts when pests are detected, ensuring timely intervention.

## Dataset
The model is trained on a robust dataset from Kaggle https://www.kaggle.com/datasets/gauravduttakiit/agricultural-pests-dataset/ , featuring various agricultural pests including:
- Ants
- Bees
- Beetles
- Caterpillars
- Earthworms
- Earwigs
- Grasshoppers
- Moths
- Slugs
- Snails
- Wasps
- Weevils

## Model Architecture
Leveraging the EfficientNetV2B0 architecture, the model has been fine-tuned specifically for pest detection. Key modifications include:
- **Freezing Base Layers:** To retain pre-trained knowledge.
- **Custom Classification Head:** Tailored for the specific pest classes.
- **L2 Regularization:** Applied to prevent overfitting.
- **Optimizer:** Adam optimizer for efficient training.

## Training Process
- **Epochs:** 5
- **Early Stopping:** Configured with a patience of 2 epochs to halt training if the validation loss does not improve.

## Real-Time Inference
The system captures live video feed through a webcam, processes each frame, and overlays pest detection results directly on the video stream. This enables farmers to monitor their fields in real-time and take immediate action when pests are detected.

## How to Set Up
>Nous recommandons l'utilisation d'une distribution Linux ou macOS.

### Step 1: Clone the Repository

> git clone https://github.com/18hyacinthe/Agricultural-Pest-Management

> cd Agricultural-Pests-management

### Step 2: Download the Pre-trained Model
Ensure you download the EfficientNetV2 pre-trained model and place it in the project directory.

### Step 3: Run the Live Monitoring Script
> python live_monitoring.py


Each frame from the video feed is processed, and the model predicts the presence of pests. The results are overlayed on the video.

# Feel Free to Contribute!

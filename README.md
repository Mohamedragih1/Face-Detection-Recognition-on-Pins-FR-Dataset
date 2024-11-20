# Face Detection and Recognition System with ResNet50 and Real-time Recognition

## Introduction
This project aims to develop a face recognition system using deep learning techniques. The project consists of three notebooks, each focusing on different aspects of the face recognition pipeline. The notebooks are:

1. **Face Detection and Cropping**: This notebook focuses on detecting faces in images using OpenCV and cropping them for further processing.
2. **Model Training**: This notebook is dedicated to training a deep learning model for face recognition using the ResNet50 architecture.
3. **Real-time Face Recognition**: This notebook demonstrates real-time face recognition using a webcam, utilizing the trained model and embeddings extracted during training.

---

## 1. Face Detection and Cropping Notebook

### Overview
This notebook extracts a dataset of images containing faces and processes them for face detection and cropping. It uses a pre-trained face detection model to detect faces in images and saves the cropped faces for further processing.

### Key Steps
- **Loading the Dataset**: The notebook loads a dataset of images containing faces. This dataset is used for face detection and cropping.
- **Face Detection**: It loads a pre-trained face detection model using OpenCV. This model detects faces in the images from the dataset.
- **Face Cropping**: Detected faces are cropped from the images and saved in a separate directory for further processing.

---

## 2. Model Training Notebook

### Overview
This notebook focuses on training a deep learning model for face recognition using the InceptionV3 architecture. It fine-tunes the pre-trained InceptionV3 model on the extracted face images and prepares it for face recognition.

### Key Steps
- **Data Preprocessing**: The extracted face images are preprocessed for training, including resizing and normalizing the images.
- **Model Architecture**: The pre-trained ResNet50 model is loaded and its layers are frozen. Additional layers are added for classification.
- **Model Training**: The model is compiled and trained using the preprocessed face images.
- **Transfer Learning**: The model is fine-tuned using transfer learning, leveraging the pre-trained ResNet50 model.
- **Model Compilation**: The model is compiled for face recognition.
- **Model Saving**: Once training is complete, the model is saved for future use.

---

## 3. Real-time Face Recognition Notebook

### Overview
This notebook demonstrates real-time face recognition using a webcam. It loads the trained face recognition model and class embeddings extracted during training. The notebook processes webcam frames for face recognition and displays the recognized faces in real-time.

### Key Steps
- **Loading the Model**: The pre-trained face recognition model is loaded from the saved file.
- **Loading Class Embeddings**: Class embeddings for each class (person) in the dataset are loaded from the saved file.
- **Face Recognition**: Webcam frames are processed for face recognition. The model predicts embeddings for each detected face, which are compared with the class embeddings using cosine similarity.
- **Cosine Similarity**: Cosine similarity measures the similarity between embeddings. If the similarity exceeds a predefined threshold, the face is recognized as belonging to a known person.
- **Real-time Display**: Recognized faces are displayed in real-time on the webcam feed.

 ## Test Cases
 ![image](https://github.com/user-attachments/assets/a845a1b8-4fe6-46f7-8b5c-d5624e4055af)

![image](https://github.com/user-attachments/assets/adadc782-26d1-4beb-a7e5-64edbaa52115)

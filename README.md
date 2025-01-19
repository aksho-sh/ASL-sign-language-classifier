# Novel Approach to Recognize and Translate Real-Time Sign Language Using Neural Networks

## **Contributors:**
- **Akshobhya Sharma**
- **Nazmus Saquib**

---

## **Project Overview**
This project presents a real-time sign language recognition system designed to translate American Sign Language (ASL) gestures using advanced neural networks. By integrating static and dynamic gesture recognition with robust preprocessing pipelines, the system bridges communication gaps for individuals with speech and hearing impairments.

## **Directory Structure**
The repository contains the following key directories:

### 1. **Dynamic_sign**
- **Contents:**
  - Trained model and a script to run the model.
  - A simple app-like program that predicts short videos as "Thank You" or "Hello" signs.
  - Includes an interactive UI with visible controls in the app window.

### 2. **Static_Sign**
- **Contents:**
  - A script and trained model to display the label of static letter signs made by the user.

### 3. **Processing**
- **Contents:**
  - Scripts used for preprocessing data.
  - Includes partial preprocessing steps (full preprocessing steps are detailed in the training notebooks).

### 4. **Training_notebooks**
- **Contents:**
  - Two Jupyter notebooks for training models.
  - Includes all preprocessing steps, training loops, testing procedures, and relevant data visualizations.

---

## **Instruction**
1. Each folder contains its own `requirements.txt` file.
2. Use the following command to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. After installing dependencies, run the script in the terminal to access the UI.

---

## **Problem Definition**
Communication barriers hinder individuals with speech and hearing impairments from engaging fully in society. Existing solutions for sign language translation lack scalability, real-time recognition capabilities, and user adaptability. This project seeks to develop a robust, real-time ASL expression translator capable of recognizing both static and dynamic gestures with high accuracy.

---

## **Datasets**
The model was trained on a combination of publicly available datasets and a custom-built dataset:

- **Static Gesture Dataset:** ~86,000 images.
- **Dynamic Gesture Dataset:**
  - "Hello": 424 videos
  - "Thank You": 472 videos
  - "Negative": 1,400 videos
- **Custom Dataset:** Built by contributors.

### Dataset Links:
1. [Custom Dataset](https://drive.google.com/file/d/1N46K7Ye-JSrlwdG-b7JqiNlaFb6LW_X2/view?usp=sharing)
2. [Kaggle Dataset 1](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out)
3. [Kaggle Dataset 2](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

---

## **Proposed Approach and Architecture**
### **1. Data Preprocessing**
- **Mediapipe:** Extracts 3D hand landmarks (21 key points per frame).
- **Normalization:** Standardizes coordinates for consistent inputs.
- **Sequence Padding:** Ensures uniform input lengths by padding or truncating videos to 128 frames.

### **2. Deep Learning Model**
- **Architecture:** Bidirectional LSTM with 3 layers, 256 hidden units per layer, dropout regularization (40%), and a fully connected classification head.
- **Training Strategy:**
  - Weighted cross-entropy loss to address class imbalance.
  - OneCycleLR scheduler for optimized learning dynamics.

---

## **Results**
### **Performance Metrics**
- **Overall Accuracy:** 97.25%
- **Precision:** 97%
- **Recall:** 97%
- **F1-Score:** 97%

### **Class-Level Metrics**
- **Hello:** Precision: 99%, Recall: 99%, F1-Score: 99%
- **Thank You:** Precision: 95%, Recall: 99%, F1-Score: 97%

### **Qualitative Observations**
- Consistent recognition across varied lighting conditions and camera angles.
- Successfully detected gestures from first-time users.

---

## **Analysis of Results**
### **Strengths**
- Robust temporal understanding from bidirectional LSTMs.
- High-quality feature extraction via Mediapipe landmarks.
- Strong performance in both static and dynamic gesture recognition.
- Generalizability across diverse users.

### **Limitations**
- Underrepresented classes (e.g., "M" and "N") showed reduced precision and recall.
- Limited dataset diversity in terms of user demographics and environmental conditions.

### **Comparison with State-of-the-Art**
The system demonstrates competitive performance, leveraging advanced preprocessing pipelines and bidirectional LSTMs for robust dynamic gesture recognition. However, state-of-the-art systems using advanced augmentation or multilingual datasets may outperform in specialized scenarios.

---

## **Citation**
1. Shivashankara, S., & Srinath, S. (2018). American Sign Language recognition system: An optimal approach. [DOI](https://doi.org/10.5815/ijigsp.2018.08.03)
2. Lee, C. K. M., Ng, K. K. H., et al. (2021). American sign language recognition with recurrent neural networks. [DOI](https://doi.org/10.1016/j.eswa.2020.114403)

---

This README provides a comprehensive overview of the project, guiding users through setup, usage, and technical details. For more information, refer to the training notebooks and individual directory contents.

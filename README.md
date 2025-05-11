# Facial Expression Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to classify facial images into seven emotional categories: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**. It uses a dataset from Kaggle and incorporates performance-boosting strategies like data augmentation, regularization, and early stopping.


## 📌 Objectives

- Develop a CNN-based classifier for facial expression recognition.
- Handle class imbalance with targeted data augmentation.
- Apply regularization (Dropout, L2) to reduce overfitting.
- Optimize model via callbacks like EarlyStopping and ReduceLROnPlateau.

## 🧠 Model Architecture

The model consists of:
- 4 Convolutional layers with ReLU activation, BatchNorm, MaxPooling, Dropout
- L2 Regularization (increasing with depth)
- Flatten + Dense(256) + Dropout(0.5)
- Output Dense layer with Softmax (7 classes)

## 📊 Results

- **Training Accuracy**: 81.51%
- **Validation Accuracy**: 66.12%
- **Best Recognized Emotion**: Happy (84.7%)
- **Most Confused Emotions**: Fear–Sad, Angry–Neutral

## 🧪 Evaluation

- Confusion Matrix & Classification Report used for analysis
- Visualization of training/validation accuracy & loss
- Insights on overfitting and augmentation impact

## 🔧 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy / Pandas / Matplotlib / Seaborn
- Google Colab (for GPU acceleration)

## ⚙️ Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/MouniReddy21/Facial-Expression-Recognition-using-CNN.git
    cd face-expression-recognition
    ```

2. (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or .\venv\Scripts\activate on Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Launch the notebook:
    ```bash
    jupyter notebook faceExpressionRecognition_q1.ipynb
    ```

## 📚 Dataset

- **Source**: [Kaggle - Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- **Preprocessing**: 
  - Resized to 48x48 pixels
  - Normalized to [0, 1]
  - Augmented to balance emotion classes

## 📈 Future Enhancements

- Integrate Attention or Transformer modules
- Explore Transfer Learning with pre-trained models
- Add temporal (video-based) emotion recognition

## 👥 Contributors

- **Mounika Seelam**
- **Prajwal Devaraj** 
- **Bhanu Prasad Dharavathu**

## 📄 License

This project is for academic and non-commercial use only.

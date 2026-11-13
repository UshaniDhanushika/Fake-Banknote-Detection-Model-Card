# Fake Banknote Detection

## 1. Project Introduction

This is a Data Science and Machine Learning project.

The main purpose of this project is to classify banknotes as genuine or forged using Machine Learning.

The UCI Banknote Authentication dataset is used to train and test the models. This project also studies model performance, limitations, and possible risks.

## 2. Problem Statement

Fake banknotes can cause financial losses. Checking banknotes manually can also take time.

Machine Learning can help classify banknotes by studying numerical measurements extracted from banknote images.

In this project, machine learning models are developed to classify banknotes as genuine or forged.

## 3. Research Question

How accurately can Machine Learning identify forged banknotes, and what are the limitations and risks of using this model?

## 4. Project Objectives

- Understand the banknote dataset.
- Explore and analyze the data.
- Prepare the data for Machine Learning.
- Build a simple baseline model.
- Train a Logistic Regression model.
- Compare model results.
- Study incorrect predictions.
- Explain the model's limitations and risks.
- Create a Responsible AI Model Card.

## 5. Dataset

**Dataset Name:** UCI Banknote Authentication Dataset

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

The dataset contains 1,372 records.

It has four numerical input features:

- Variance
- Skewness
- Curtosis
- Entropy

These features are measurements extracted from banknote images using a Wavelet Transform.

The target column is called `class`:

- `0` - Genuine banknote
- `1` - Forged banknote

## 6. Tools and Technologies

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- GitHub

## 7. Machine Learning Models

Two models are used in this project:

1. **Dummy Classifier** - Used as a simple baseline to compare model performance.
2. **Logistic Regression** - Used to classify banknotes as genuine or forged.

The results of both models will be compared.

## 8. Model Evaluation

The following methods are used to evaluate the models:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Cross-validation
- Error Analysis

## 9. Responsible AI

This project studies the limitations and risks of the machine learning model.

It also explains the possible problems caused by incorrect predictions, such as classifying a forged banknote as genuine or a genuine banknote as forged.

This model is developed for educational purposes only. It should not be used as a real banknote verification system.

## 10. Project Structure

```text
fake-banknote-detection-model-card/
│
├── README.md
├── notebooks/
│   └── fake_banknote_detection.ipynb
├── reports/
│   └── model_card.md
├── images/
└── requirements.txt
```

## 11. Project Status

The project is currently in development.

## 12. Author

Created as a Data Science examination project.

# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using machine learning techniques. It utilizes various models such as Neural Networks, Random Forest Classifier, and Logistic Regression to predict survival based on features like age, sex, ticket class, and more.

## Installation

To run this project, you need Python installed on your system. Clone the repository and install the required dependencies using pip:

```bash
git clone https://github.com/mdokusV/ML-python.git
cd zaj10
pip install -r requirements.txt
```

## Usage

To use this project:

1. Ensure you have the necessary dataset files: \`train.csv\`, \`test.csv\`, and \`gender_submission.csv\`.
2. Update the file paths in the script if necessary.
3. Run the script:

```bash
python best_model_keras.h5
```

## Features

- Data preprocessing: Handling missing values, encoding categorical features, and feature engineering.
- Model training: Utilizes various machine learning models like Neural Networks, Random Forest Classifier, and Logistic Regression.
- Evaluation: Evaluates models based on metrics such as F1-score, Accuracy, Precision, and Recall.
- Special Output (Optional): Provides detailed analysis including predictions and raw values for better understanding.

## Model Performance

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| Neural Network         | 0.8582   | 0.8700    | 0.7838 | 0.8242   |
| RandomForest Classifier| 0.8314   | 0.8335    | 0.8314 | 0.8291   |
| Logistic Regression    | 0.8621   | 0.8633    | 0.8621 | 0.8608   |

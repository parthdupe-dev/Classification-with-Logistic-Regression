# Task 4: Classification with Logistic Regression

## Objective
Build a binary classification model using Logistic Regression to classify breast cancer cases as Benign or Malignant.

## Dataset
- Dataset: Breast Cancer Wisconsin Dataset
- Total Records: 569
- Features: 30
- Target: Diagnosis
- Benign (B): 0
- Malignant (M): 1

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Implementation
1. Loaded and inspected the dataset.
2. Removed unnecessary columns (`id` and `Unnamed: 32`).
3. Encoded the target variable.
4. Split the dataset into training and testing sets (80:20).
5. Standardized numerical features using StandardScaler.
6. Trained a Logistic Regression model.
7. Generated predictions and class probabilities.
8. Evaluated the model using a confusion matrix, classification report, and ROC-AUC.
9. Demonstrated threshold tuning using a threshold of 0.3.
10. Generated an ROC curve.

## Model Evaluation

### Default Threshold (0.5)
- Accuracy: 96%
- Malignant Precision: 0.97
- Malignant Recall: 0.93
- Malignant F1-score: 0.95
- ROC-AUC: 0.9960

### Tuned Threshold (0.3)
- Accuracy: 98%
- Malignant Precision: 0.98
- Malignant Recall: 0.98
- Malignant F1-score: 0.98

## Confusion Matrix (Default Threshold)
```text
[[71  1]
 [ 3 39]]
```

## Project Structure
```text
DAY_TASK_4/
├── data.csv
├── main.py
└── README.md
```

## How to Run
1. Install the required libraries:
   `pip install pandas numpy scikit-learn matplotlib seaborn`
2. Run:
   `python3 main.py`

## Note
This project is developed for educational purposes and is not intended for clinical diagnosis.

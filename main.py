import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load and Prepare the Dataset
# --------------------------------------------------

df = pd.read_csv('data.csv')


# Remove non-predictive identifier and completely empty columns.
df = df.drop(columns=['id',
                     'Unnamed: 32'
                     ])

'''print(df.shape)
print(df.head())
print(df.columns)

print(df.select_dtypes(include=['number']).columns.tolist())
print(df.select_dtypes(exclude=['number']).columns.tolist())

has_missing = df.isna().sum()
print("missing values\n", has_missing)

counts = df['diagnosis'].value_counts()

print(counts)'''


# --------------------------------------------------
# 2. Separate Features and Target Variable
# --------------------------------------------------

# Define independent features (X) and the dependent target variable (Y).
X = df.drop(columns=['diagnosis'])
Y = df[['diagnosis']]

# Convert categorical diagnosis labels into numerical values:
# 0 represents Benign and 1 represents Malignant.
Y['diagnosis'] = Y['diagnosis'].map({'B': 0, 'M': 1})

"""
print(Y.head())
print(Y.value_counts())
"""


# --------------------------------------------------
# 3. Split Dataset into Training and Testing Sets
# --------------------------------------------------

# Allocate 80% of the data for training and 20% for testing.
# Stratification preserves the original class distribution in both subsets.
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)


# --------------------------------------------------
# 4. Feature Scaling
# --------------------------------------------------

# Initialize the StandardScaler to normalize feature distributions.
scaler = StandardScaler()

# Fit the scaler using training data to prevent data leakage.
scaler.fit(X_train)

# Apply standardization to training and testing features.
# The same training-fitted scaling parameters are used for both datasets.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''
print(X_train_scaled.shape)
print(X_test_scaled.shape)
print(X_train_scaled.mean())
print(X_test_scaled.std())
'''


# --------------------------------------------------
# 5. Initialize and Train the Logistic Regression Model
# --------------------------------------------------

# Initialize Logistic Regression with an increased iteration limit
# to provide sufficient iterations for model convergence.
model = LogisticRegression(max_iter=1000)

# Train the classification model using standardized training features.
model.fit(X_train_scaled, y_train.values.ravel())


# --------------------------------------------------
# 6. Generate Model Predictions
# --------------------------------------------------

# Generate binary class predictions for the test dataset.
y_pred = model.predict(X_test_scaled)

# Generate class probabilities for each test observation.
y_pred_proba = model.predict_proba(X_test_scaled)


# --------------------------------------------------
# 7. Evaluate Model Performance
# --------------------------------------------------

# Confusion Matrix:
# Evaluate the distribution of correct and incorrect predictions.
# Rows represent actual classes; columns represent predicted classes.
# Class 0: Benign | Class 1: Malignant.
# Matrix layout:
# [[True Negatives, False Positives],
#  [False Negatives, True Positives]]
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Classification Report:
# Calculate class-wise precision, recall, F1-score, and support.
# Precision measures the reliability of positive predictions.
# Recall measures the proportion of actual positive cases identified.
# F1-score represents the harmonic mean of precision and recall.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))


# --------------------------------------------------
# 8. Calculate ROC-AUC Score
# --------------------------------------------------

# Calculate the Area Under the Receiver Operating Characteristic Curve.
# ROC-AUC measures the model's ability to distinguish between classes.
# A score closer to 1.0 indicates stronger class discrimination.
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print(f"\nROC-AUC Score: {roc_auc:.4f}")


# --------------------------------------------------
# 9. Visualize the ROC Curve
# --------------------------------------------------

# Generate the ROC curve using the trained model and test dataset.
# The curve illustrates the trade-off between True Positive Rate
# and False Positive Rate across different classification thresholds.
print("\nPlotting ROC Curve...")
disp = RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
disp.ax_.set_title('ROC Curve for Logistic Regression')
plt.show()


# --------------------------------------------------
# 10. Classification Threshold Optimization
# --------------------------------------------------

# Logistic Regression estimates class probabilities rather than
# directly producing binary classifications.
#
# The sigmoid function converts the model's linear output into
# a probability value between 0 and 1:
# σ(z) = 1 / (1 + e^(-z))
#
# By default, probabilities >= 0.5 are classified as class 1,
# while probabilities below 0.5 are classified as class 0.
#
# Adjusting the decision threshold changes the classification
# sensitivity and the balance between precision and recall.

print("\n--- Threshold Tuning Demonstration ---")


# Extract predicted probabilities for the positive class (Malignant).
y_prob = model.predict_proba(X_test_scaled)[:, 1]


# Define an alternative classification threshold for comparison.
# A lower threshold makes the classifier more likely to predict
# the positive class, potentially increasing recall.
new_threshold = 0.3

# Convert predicted probabilities into binary class predictions
# using the newly defined decision threshold.
y_pred_new_threshold = (y_prob >= new_threshold).astype(int)


# Evaluate model performance using the adjusted threshold.
conf_matrix_new = confusion_matrix(y_test, y_pred_new_threshold)
print(f"Confusion Matrix with a new threshold of {new_threshold}:")
print(conf_matrix_new)


# Generate the classification report to compare precision,
# recall, F1-score, and support after threshold adjustment.
#
# Lowering the threshold may increase True Positives and recall,
# but can also increase False Positives and reduce precision.
print("\nClassification Report with a new threshold of 0.3:")
print(classification_report(y_test, y_pred_new_threshold, target_names=['Benign', 'Malignant']))
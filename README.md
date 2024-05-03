# Human-Activity-with-Smartphones
## SVM-PCA for Human Activity Recognition

This project implements a Support Vector Machine (SVM) classifier with Principal Component Analysis (PCA) for dimensionality reduction. The classifier is trained on the Human Activity Recognition Using Smartphones Dataset to recognize the type of activity performed by a person based on smartphone sensor data.

## About the Dataset:
The Human Activity Recognition Using Smartphones Dataset contains data collected from the accelerometers and gyroscope of smartphones carried by volunteers. It includes 561 features extracted from the smartphone sensor data and the activity labels for each data point.

## Purpose:
The purpose of this project is to classify human activities based on smartphone sensor data using machine learning techniques. By implementing SVM with PCA for dimensionality reduction, we aim to achieve high classification accuracy while reducing the computational complexity.

## Steps Involved:
1. **Data Loading and Exploration:** 
   - Load the dataset and explore its structure.
   - Analyze the distribution of the target variable and some of the numerical features.
   - Visualize the correlation between features using a heatmap.

2. **Data Preprocessing:**
   - Separate the features and the target variable.
   - Standardize the features using `StandardScaler`.
   - Apply PCA for dimensionality reduction.

3. **Model Development:**
   - Split the data into training and testing sets.
   - Define the SVM model.
   - Use RandomizedSearchCV for hyperparameter tuning.
   - Train the SVM model on the training data.
   - Make predictions on the test data.
   - Evaluate the model performance using accuracy and classification report.

## Contents:
- `SVM_PCA.ipynb`: Jupyter Notebook containing the SVM-PCA implementation.
- `Human_Activity_Recognition_Using_Smartphones_Data.csv`: Dataset used for training the SVM-PCA model.

## Dependencies:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Usage:
1. Clone this repository:

```bash
git clone https://www.kaggle.com/code/khaledsamy/svm-pca.git

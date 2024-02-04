Breast Cancer Wisconsin (Diagnostic) Data Analysis

Introduction

This project focuses on the analysis and prediction of breast cancer diagnosis using the Breast Cancer Wisconsin (Diagnostic) Dataset. The goal is to develop machine learning models that can accurately classify tumors as benign or malignant based on features derived from digitized images of fine needle aspirates (FNA) of breast masses.

Models Description

Two models were developed for this project:

Model V1: Logistic Regression
Algorithm: Logistic Regression
Purpose: To provide a baseline model for binary classification of breast cancer tumors.
Key Features:
Uses 30 input features including mean, standard error, and worst values of cell nucleus characteristics.
Standardized feature set for optimal performance.
Evaluated using accuracy and confusion matrix.
Model V2: Support Vector Machine (SVM) with RBF Kernel
Algorithm: Support Vector Machine with Radial Basis Function (RBF) kernel
Purpose: To improve classification performance over the baseline model by handling non-linear relationships.
Key Features:
Employs the same 30 input features as Model V1.
Includes parameter tuning for C and gamma to optimize decision boundaries.
Evaluated using the same metrics as Model V1 for consistency.
How to Use

Prerequisites
Python 3.x
Jupyter Notebook or JupyterLab
Pandas
Scikit-learn
Setup and Running Models
Clone the repository to your local machine.
Install the required dependencies using pip install -r requirements.txt.
Open the Jupyter Notebook for the desired model (model_v1.ipynb or model_v2.ipynb).
Run each cell in the notebook to train the model and view the results.
Results

The models were evaluated based on their accuracy and the confusion matrix. Here are the summarized results:

Model V1 (Logistic Regression): Achieved an accuracy of approximately 97.37%, with a very low false positive rate.
Model V2 (SVM with RBF Kernel): Showed a slight improvement with an accuracy of 98.25%, and no false positives in the test set.
Conclusion

Both models demonstrated high accuracy in classifying breast cancer tumors. The SVM model with RBF kernel provided a marginal improvement in accuracy and specificity, making it a strong candidate for further development and tuning.

Future work may include exploring more advanced algorithms, feature selection techniques to reduce model complexity, and cross-validation to ensure model robustness.

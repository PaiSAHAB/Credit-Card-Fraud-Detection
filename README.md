# Fraud Detection Project

## Overview
This project builds supervised and unsupervised models to detect fraudulent credit card transactions using a publicly available dataset.

## Steps
1. **Data Exploration & Preprocessing:**  
   - Loaded the credit card transactions dataset.
   - Scaled numerical features (Time, Amount).
   - Addressed class imbalance using SMOTE.

2. **Supervised Modeling:**
   - Baseline model: Logistic Regression.
   - Advanced model: XGBoost with basic hyperparameter tuning.
   - Evaluated using accuracy, recall, precision, F1-score, confusion matrix, and ROC/PR curves.

3. **Explainability:**
   - Extracted feature importance from XGBoost.
   - Used SHAP for model explainability, providing both global and local interpretations.


Let us build our focus more towards the unsupervised approach.

## Fraud Detection Using Unsupervised Approaches

## Overview
This project demonstrates the use of unsupervised anomaly detection methods to identify potential fraudulent credit card transactions. Two primary approaches are showcased:

1. **Isolation Forest:** A tree-based method that isolates anomalies by randomly splitting features.
2. **Autoencoder (Neural Network):** A deep learning model trained only on normal transactions to learn their pattern and use reconstruction error to flag anomalies.

The dataset used is the commonly available Credit Card Fraud Detection dataset from Kaggle.

## Key Objectives
- Highlight unsupervised modeling techniques for fraud detection.
- Show how to evaluate and interpret anomaly detection results.
- Compare the performance of different approaches and provide insights into their behavior and practicality.

## Model Performance Summary

### Isolation Forest
- **Method:** Tree ensemble that isolates anomalies.
- **Contamination Parameter:** Set to a small fraction to indicate expected proportion of anomalies.
- **Performance Metrics:**  
  - Classification Report (compared against known labels):
    - Shows precision, recall, and F1-score for both normal and fraudulent classes.
  - Confusion Matrix:
    - Visualizes how many fraudulent cases were correctly identified vs. missed.
  - ROC & PR Curves:
    - Display the trade-off between the true positive rate and false positive rate (ROC), and precision vs. recall (PR).

**Key Insight:**  
The Isolation Forest can detect a subset of anomalies without supervision. Adjusting the contamination parameter is critical. The method is fast and straightforward but may require domain knowledge to tune its hyperparameters effectively.

### Autoencoder
- **Method:** Neural network trained exclusively on normal (non-fraudulent) transactions.
- **Reconstruction Error as a Metric:**
  - The model tries to reconstruct normal transactions accurately.  
  - Fraudulent transactions often have higher reconstruction errors due to their unusual patterns.
- **Threshold Setting:**  
  - Chosen based on a percentile of reconstruction errors on the normal set (e.g., 99.5th percentile).
- **Performance Metrics:**
  - Classification Report: Showcases how well the chosen threshold separates fraud from normal.
  - Confusion Matrix: Summarizes correct and incorrect anomaly detections.
  - ROC & PR Curves: Derived from reconstruction errors to measure how well increasing the threshold can separate anomalies from normal data.

**Key Insight:**  
The Autoencoder provides a flexible, learned representation of normal behavior. By focusing on reconstruction error, we can adapt the model as patterns evolve. Fine-tuning the architecture, epochs, and thresholding strategy can improve detection rates.

## Visualizations and Plots
- **Confusion Matrices (Isolation Forest & Autoencoder):**  
  Displayed to illustrate the count of true positives, false negatives, etc. Helps quickly identify the model’s ability to detect fraud.
  
- **ROC Curves (Isolation Forest & Autoencoder):**  
  Show how varying the threshold affects the true positive rate (sensitivity) and false positive rate. Helps compare model performance at different operating points.

- **Precision-Recall Curves (Isolation Forest & Autoencoder):**  
  Particularly useful in highly imbalanced datasets. Display how precision and recall change with different thresholds and help understand performance when the positive class (fraud) is rare.

- **Reconstruction Error Distributions (Autoencoder):**  
  Histograms illustrating the difference in error distributions for normal vs. fraudulent transactions.  
  - Normal transactions typically cluster around low reconstruction errors.
  - Fraudulent transactions often appear in the long tail of high reconstruction errors.


### Examples of Detected Anomalies
- By applying the trained models to the dataset, certain transactions with high anomaly scores (Isolation Forest) or high reconstruction errors (Autoencoder) are flagged.  
- Comparison with the known labels (not used during training) shows whether these flagged transactions align with actual fraud cases.  
- This evaluation highlights that while not perfect, the unsupervised models capture a significant portion of fraudulent cases without any prior labeling.

## Getting Started

1. **Prerequisites:**
   - Python 3.x
   - Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`/`keras`

2. **Data:**
   - Download the credit card fraud dataset from Kaggle and place `creditcard.csv` in the working directory.

3. **How to Run:**
   - Open the Jupyter Notebook or Python script provided.
   - Run through the cells to load data, train models, and produce evaluation metrics and plots.

## Results and Insights
- Unsupervised methods detect anomalies without labeled training.
- Isolation Forest provides a fast baseline but may need tuning for optimal sensitivity.
- Autoencoders offer a more flexible, learnable model of normal behavior, potentially improving detection with more careful tuning.
- Combining insights from both approaches can help build a robust anomaly detection system that adapts over time.

---

This README provides an overview of the unsupervised fraud detection project, outlines key performance metrics, visualizations, and results, and explains how to reproduce and interpret the findings.

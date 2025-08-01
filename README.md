# Find Out Your Personality Project

## Overview
   This application uses a machine learning algorithm (Support Vector Machine, SVM) to predict a user’s personality type based on input traits. The app leverages Streamlit for an interactive experience, allowing users to explore the underlying dataset, check model accuracy, input their own trait values, see predictions, and download the data all in one place.

## Features
   Interactive Input: Enter new trait values (range 1 to 10) on the sidebar to get instant personality predictions.
   Dataset Display: View the first five rows of the personality synthetic dataset.
   Model Accuracy: See how well the algorithm performs on test data.
   Download Option: One-click to download the entire dataset.
   Classification Report: Quickly assess model performance with standard classification metrics.

## How It Works
### Dataset Loading: 
    The app loads a CSV dataset (personality_synthetic_dataset.csv) containing various traits and labeled personality types.

### Data Preprocessing: 
    Checks for missing or duplicate data.
    Features are standardized using a StandardScaler to improve model performance.

### Model Training:
    The features (X) and target labels (y) are split into training and test sets.
    A Support Vector Classifier (SVC) with an RBF kernel is trained on the training set.

### Evaluation:
  The model predicts personality types for the test set.
  Accuracy and a classification report are presented.

### User Prediction:
    Users input their trait values (all traits in the range 1.0 to 10.0).
    These values are scaled and fed to the classifier, which predicts the user's personality type.

## Requirements

  Python 3.7+
  Libraries: streamlit, pandas, numpy, scikit-learn
  The dataset file: personality_synthetic_dataset.csv (should be present in the specified directory)

## How to Run

To run this project run the below command in terminal.
streamlit run personality.py

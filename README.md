# Industrial-Copper-Modeling-Selling-price-and-Status-Prediction-using-ML
This project includes Machine Learning models for predicting the selling price and status of industrial copper transactions. Additionally, it features a Streamlit web application for interactive prediction using trained models.

## Table of Contents
1. Project Overview
2. Installation
3. Usage
4. File Description
5. Models Trained
6. Saving Model
7. Streamlit Application
8. Results
9. Project Overview

## In this project, we trained Machine Learning models to predict:

Selling Price of industrial copper based on various features.
Status (Won or Lost) of copper transactions.

The models were developed using Python, and the Streamlit framework was used to create an interactive web application for users to make predictions.

## Installation
To run this project locally and use the Streamlit app, follow these steps:
Libraries to import and install:

import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

## Usage
Data Preprocessing: The dataset is cleaned, preprocessed, and used to train Machine Learning models.
Model Training: Models for predicting selling price (regression) and status (classification) are trained using algorithms such as Random Forest, Gradient Boosting, and others.
Streamlit Application: Use the Streamlit web application to interactively make predictions based on user inputs.
Model Evaluation: Model performance metrics are evaluated and displayed in the application.

## File Descriptions
industrial_copper_modeling.ipynb: Colab notebook containing the entire project code, including model training and evaluation.
Extra_trees.pkl: Serialized Extra Trees classification model.
stream.py: Python script containing the Streamlit application code (stream.py).

## Models Trained

Regression Models:

Random Forest Regression
Gradient Boosting Regression
Linear Regression
Decision Tree Regressor

Classification Models:

Extra Trees Classifier
Random Forest Classifier
KNeighbors Classifier
Gradient Boosting Classifier
Logistic Regression
Decision Tree Classifier

## Saving Model

Save the models that have high acurricy in prediction as a .pkl file. 

## Streamlit Application

The Streamlit application (stream.py) allows users to:

Predict the selling price of copper based on input features such as item date, quantity, country, etc.
Predict the status (Won or Lost) of copper transactions based on input features including item date, selling price, customer details, etc.
To run the Streamlit app:

## Results

Regression Results:

Random Forest Regression showed the best performance in predicting the selling price based on evaluation metrics like RMSE and R².

Classification Results:

Extra Trees Classifier achieved high accuracy in predicting transaction status (Won or Lost).

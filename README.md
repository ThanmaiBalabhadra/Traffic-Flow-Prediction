# 🚦 Accident Severity Prediction Using Machine Learning

This project focuses on predicting the **severity of traffic accidents** using machine learning algorithms. By analyzing real-world traffic accident data, the system aims to identify patterns and predict whether an accident will result in minor, serious, or fatal outcomes.

## Project Title

**Predictive Modelling of Machine Learning Algorithms for Traffic Flow in Smart Transportation System**

---

## Table of Contents

- [About](#about)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Tools and Technologies](#tools-and-technologies)
- [ML Models Used](#ml-models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)
- [Acknowledgements](#acknowledgements)

---

## About

The system predicts the **severity of road accidents** based on various input factors using machine learning models. The goal is to enhance **smart transportation systems** by enabling proactive measures to reduce accident impacts.

---

## Problem Statement

Accurate prediction of accident severity can help in efficient allocation of emergency resources and improve traffic safety planning. Existing systems lack high-accuracy, real-time predictive capabilities. This project aims to fill that gap using ML techniques.

---

## Objectives

- Analyze the dataset for feature relevance using a correlation matrix.
- Train multiple ML algorithms to predict accident severity.
- Compare models based on performance metrics.
- Visualize results through confusion matrix, heatmap, and bar charts.

---

## Dataset

- **Source:** Kaggle
- **Preprocessing:** Null values removed (dataset had none initially)
- **Features:** Include weather, road conditions, vehicle type, special conditions, etc.

> **Note:**  
> - `Vehicle Type`: 1 = Lorry, 0 = Non-Lorry  
> - `Special Conditions`: 1 = Present, 0 = Not Present

---

## Tools and Technologies

- **Programming Language:** Python  
- **Libraries:** pandas, numpy, seaborn, matplotlib, scikit-learn  
- **ML Models:** Random Forest, Decision Tree, SVM, Logistic Regression  
- **Visualization:** Confusion Matrix, Heatmap, Accuracy Bar Graph  

---

## ML Models Used

- ✅ **Random Forest Classifier** *(Best Accuracy)*
- ✅ **Decision Tree**
- ✅ **Support Vector Machine (SVM)** - Linear Kernel
- ✅ **Logistic Regression**

---

## Evaluation Metrics

- Accuracy ✅
- Precision
- Recall
- F1-Score

> ✔️ Focus on **Accuracy** as the primary metric in this project

---

## Visualizations

- **Correlation Matrix (Heatmap)** - For feature selection
- **Confusion Matrix** - For prediction analysis
- **Bar Graph** - Comparing model accuracies

---

## Results

- **Random Forest** achieved the highest accuracy: **95%**
- Precision, Recall, and F1-score: **~94–95%**
- Ensemble methods perform best due to the non-linear nature of the dataset

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/ThanmaiBalabhadra/Traffic-Flow-Prediction.git
   cd Traffic-Flow-Prediction

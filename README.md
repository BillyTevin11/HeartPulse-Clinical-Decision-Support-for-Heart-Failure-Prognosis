# HeartPulse AI: Clinical Decision Support for Heart Failure Prognosis


## 📋 Executive Summary

Heart failure is a critical cardiovascular condition where the heart cannot pump blood effectively. This project implements a machine learning pipeline to predict mortality risk in heart failure patients using 12 clinical features. Beyond binary classification, the system provides Local Explainability, identifying the specific clinical drivers (e.g., Serum Creatinine, Ejection Fraction) contributing to an individual patient’s risk score.



## 🛠️ Technical Stack

- Data Analysis: Pandas, NumPy, Scipy (Log transformations for skewed distributions).

- Machine Learning: Scikit-Learn (Random Forest Classifier).

- Deployment: Flask API (RESTful architecture).

- Frontend: HTML5/Tailwind CSS (Responsive Medical Dashboard).

- Serialization: Joblib (Model and Scaler persistence).


## 📉 Data Science Pipeline

1. Feature Engineering & Preprocessing

- Skewness Correction: Features like serum_creatinine and creatinine_phosphokinase exhibited high right-skewness and were normalized using log transformations

- Feature Scaling: Standardized numerical features using StandardScaler to ensure the Random Forest algorithm maintains convergence stability.

- Class Imbalance: Addressed the minority class (DEATH_EVENT) using balanced_subsample class weighting.


2. Model Optimization (Recall-Focused)

In clinical diagnostics, Recall (Sensitivity) is prioritized over Accuracy to minimize "False Negatives"—high-risk patients overlooked by the system.

- Hyperparameter Tuning: Conducted via GridSearchCV optimizing for the recall metric.
    
- Threshold Engineering: Adjusted the classification threshold from $0.5$ to $0.35$, increasing Recall from 63% to 79%.


3. Local Explainability (XAI)

The system calculates feature impact by comparing patient inputs against clinical medians. This transparency allows clinicians to see which vitals are "Pushing" the risk score higher (Red) or "Protecting" the patient (Green).



## 📂 Project Structure
├── app.py                # Flask API & Backend Logic

├── model.pkl             # Trained Random Forest Model

├── scaler.pkl            # Trained StandardScaler

├── templates/index.html        # Tailwind CSS Dashboard

├── model.py              # Exploratory Data Analysis & Training

├── data.csv              # Dataset



## 📊 Results & Insights

Primary Drivers: Time (follow-up period), Serum Creatinine, and Ejection Fraction were identified as the most significant predictors of mortality.

Performance:

a. Accuracy: 85%

b. Recall (Sensitivity): 79% (at 0.35 threshold)

c. Precision: 60%




### ⚖️ Disclaimer
This tool is intended for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

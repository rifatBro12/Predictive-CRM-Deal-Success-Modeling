# CRM Sales Deal Prediction and Win Probability Forecasting

## 📌 Overview
This project builds a **machine learning system to analyze CRM sales data** and:  
1. Predict whether a sales deal will be **Won or Lost** (Classification).  
2. Forecast the **time required to close a deal** (Regression, optional extension).  

By leveraging historical CRM data (accounts, products, sales teams, and pipeline records), the model helps improve decision-making, prioritize deals, and optimize business development strategies.

---

## 📊 Dataset
The dataset is extracted from a CRM system (`crm_sales_dataset`) with the following tables:

- **accounts.csv** → 85 rows, 7 columns (company/customer information)  
- **products.csv** → 7 rows, 3 columns (product details)  
- **sales_pipeline.csv** → 8,800 rows, 8 columns (deal stages, outcomes, timelines)  
- **sales_teams.csv** → 35 rows, 3 columns (team and rep assignments)  

---

## 🎯 Objectives
- Merge and clean multiple CRM datasets.  
- Encode categorical variables (products, accounts, sales reps).  
- Train **classification models** for deal outcome prediction.  
- Train **regression models** to predict days-to-close (optional).  
- Evaluate performance using **Accuracy, Precision, Recall, F1-score, MAE, RMSE, R²**.  
- Provide actionable business insights for sales teams.  

---

## 🛠 Tech Stack
- **Python 3.x**  
- **Pandas, NumPy** (data processing)  
- **Scikit-learn** (ML models, preprocessing)  
- **Matplotlib / Seaborn** (visualizations)  
- **Jupyter Notebook** (experimentation)  

---

## 🔄 Workflow
1. **Data Loading & Extraction**  
   - Unzip the dataset and load CSV files into Pandas.  

2. **Data Preprocessing**  
   - Merge datasets into a single master table.  
   - Handle missing values and encode categorical variables.  
   - Feature engineering (account age, engagement date features, log transformations).  

3. **Modeling**  
   - **Random Forest Classifier** → Predict Deal Outcome (Won/Lost).  
   - **Optional:** Random Forest Regressor → Predict Days-to-Close.  

4. **Evaluation**  
   - Classification metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.  
   - Regression metrics: MAE, RMSE, R².  

5. **Visualization & Insights**  
   - Feature importance plots to identify key factors influencing deal success.  

---

## 📈 Baseline Results
- **Random Forest Classifier** → Predicting Win/Loss  
  - Accuracy: ~0.60 (baseline, can be improved)  
  - ROC-AUC: ~0.54  

- **Random Forest Regressor** → Predicting Days-to-Close  
  - MAE: 29.01 days  
  - RMSE: 36.03 days  
  - R²: 0.2428  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crm-sales-forecasting.git
   cd crm-sales-forecasting

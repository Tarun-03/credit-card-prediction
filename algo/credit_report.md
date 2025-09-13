# Credit Card Approval - Model Report

Generated: 2025-09-13 10:13:35

## Overview
Credit scorecards are widely used in the financial industry as a risk control measure. These cards utilize personal information and data provided by credit card applicants to assess the likelihood of potential defaults and credit card debts in the future. Based on this evaluation, the bank can make informed decisions regarding whether to approve the credit card application. Credit scores provide an objective way to measure and quantify the level of risk involved.

Credit card approval is a crucial process in the banking industry. Traditionally, banks rely on manual evaluation of creditworthiness, which can be time-consuming and prone to errors. However, with the advent of Machine Learning (ML) algorithms, the credit card approval process has been significantly streamlined. Machine Learning algorithms have the ability to analyze large volumes of data and extract patterns, making them invaluable in credit card approval. By training ML models on historical data that includes information about applicants, their financial behavior, and credit history, banks can predict creditworthiness more accurately and efficiently.

AI in the Prediction: Artificial intelligence plays a transformative role in credit scoring. Traditional credit scoring models often fail to account for the complexity and variability of individual financial behaviors. AI, on the other hand, can process vast amounts of data, identify patterns, and make predictions with a high degree of accuracy. This allows for a more personalized and fair assessment of creditworthiness. AI credit scoring also has the potential to extend credit opportunities to underserved populations, such as those with thin credit files or those who are new to credit, by considering alternative data in the scoring process.

## Key Metrics

- ROC-AUC: **1.0000**

- Classification Report:


```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      6400
           1       1.00      1.00      1.00      6400

    accuracy                           1.00     12800
   macro avg       1.00      1.00      1.00     12800
weighted avg       1.00      1.00      1.00     12800

```

## Top Feature Importances (Top 10)


````
Credit_History_Length                                602
YEARS_EMPLOYED                                       458
FLAG_OWN_REALTY                                      454
AGE                                                  430
Count_Late_Payments                                  410
Percentage_On_Time_Payments                          398
AMT_INCOME_TOTAL                                     396
NAME_EDUCATION_TYPE_Secondary / secondary special    345
NAME_INCOME_TYPE_Working                             339
NAME_INCOME_TYPE_Pensioner                           315
````

## Plots

- Model Performance: ![Model Performance](credit_analysis_model.png)

- Demographics: ![Demographics](credit_analysis_demographics.png)

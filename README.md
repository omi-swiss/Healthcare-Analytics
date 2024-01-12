# Healthcare Analytics: Patient Length of Stay Prediction

## Overview

This project focuses on using healthcare analytics to predict the Length of Stay (LOS) for patients in healthcare facilities. By analyzing LOS, we aim to improve patient outcomes, optimize hospital resources, and reduce healthcare costs.

## Objectives

- Analyze patient data using Snowflake.
- Build a Machine Learning model to predict LOS.
- Utilize AWS Sagemaker for streamlined data processing and machine learning workflows.
- Perform live data scoring and update Snowflake with predictions.
- Send automated status emails.

## Approach

1. **Utilizing Snowflake**: Leveraging Snowflake for efficient data storage, management, and exploratory data analysis (EDA).
2. **AWS Sagemaker Integration**: Employing AWS Sagemaker for a seamless and efficient machine learning pipeline.
3. **Data Fetching and Processing**: Utilizing `snowflake-connector-python` and `snowflake-sqlalchemy` for data retrieval and preprocessing.
4. **Feature Engineering and Selection**: Enhancing and selecting relevant data features for the predictive model.
5. **Model Development**: Building predictive models including Linear Regression, Random Forest Regression, and XGBoost Regression.
6. **Model Predictions and Analysis**: Generating patient LOS predictions and analyzing their accuracy.
7. **Snowflake Integration for Predictions**: Inserting model predictions back into Snowflake for real-time data updates.

## Data

- **Training Data**: Data on 230K patients with 19 features.
- **Simulation Data**: Prediction data for 71K patients.

## Tech Stack

- **Tools**: AWS Sagemaker, Snowflake.
- **Programming Language**: Python.
- **Libraries**: `snowflake-connector-python`, `snowflake-sqlalchemy`, `xgboost`, `pandas`, `numpy`, `scikit-learn`.


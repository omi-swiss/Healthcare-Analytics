

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn import metrics
import numpy as np
import joblib  # Import joblib to save the best model

def train_and_save_best_model(x_train, y_train, x_test, y_test, save_path):
    # Define a list of regression models to evaluate
    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor()),
        ("XGBoost Regressor", xgb.XGBRegressor()),
    ]

    best_model_name = ""
    best_rmse = float("inf")

    # Loop through each model and evaluate its performance
    for model_name, model in models:
        # Fit the model to the training data
        model.fit(x_train, y_train)
        
        # Calculate the R-squared score on the training data
        train_r2 = model.score(x_train, y_train)
        
        # Make predictions on the test data
        preds = np.ceil(model.predict(x_test))
        
        # Calculate RMSE and MAE
        rmse = np.sqrt(metrics.mean_squared_error(y_test, preds))
        mae = np.sqrt(metrics.mean_absolute_error(y_test, preds))
        
        # Check if this model has the lowest RMSE so far
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name

        # Print model evaluation results
        print(f"Model: {model_name}")
        print(f"Train R-squared: {train_r2:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print()
    
    # Save the best model to a file using joblib
    best_model = [model for model_name, model in models if model_name == best_model_name][0]
    joblib.dump(best_model, save_path)
    
    print(f"The best model is: {best_model_name} with RMSE: {best_rmse:.2f}")
    print(f"The best model has been saved to {save_path}")


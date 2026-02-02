"""
Machine Learning Models for Demand Prediction
Trains and evaluates multiple regression models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class MLModels:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Train multiple models and compare performance
        
        Args:
            X: Feature DataFrame
            y: Target variable (demand_score)
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Dictionary with model results
        """
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # 1. Linear Regression (Baseline)
        print("Training Linear Regression...")
        lr_results = self._train_linear_regression(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        results['Linear Regression'] = lr_results
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf_results = self._train_random_forest(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        results['Random Forest'] = rf_results
        
        # 3. XGBoost
        print("Training XGBoost...")
        xgb_results = self._train_xgboost(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        results['XGBoost'] = xgb_results
        
        # Select best model
        self._select_best_model(results)
        
        # Save best model
        self._save_model()
        
        return results
    
    def _train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model': model,
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'predictions': y_pred_test
        }
        
        self.models['Linear Regression'] = model
        
        return results
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest with hyperparameter tuning"""
        # Base model
        rf = RandomForestRegressor(random_state=42)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        # Best model
        model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model': model,
            'best_params': grid_search.best_params_,
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(self.feature_names, model.feature_importances_)),
            'predictions': y_pred_test
        }
        
        self.models['Random Forest'] = model
        
        return results
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost with hyperparameter tuning"""
        # Base model
        xgb = XGBRegressor(random_state=42, verbosity=0)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            xgb, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        # Best model
        model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model': model,
            'best_params': grid_search.best_params_,
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(self.feature_names, model.feature_importances_)),
            'predictions': y_pred_test
        }
        
        self.models['XGBoost'] = model
        
        return results
    
    def _select_best_model(self, results):
        """Select best model based on test R² score"""
        best_r2 = -np.inf
        best_name = None
        
        for name, result in results.items():
            if result['test_r2'] > best_r2:
                best_r2 = result['test_r2']
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"
✅ Best Model: {best_name} (Test R² = {best_r2:.4f})")
    
    def predict_demand(self, features):
        """
        Predict demand using best model
        
        Args:
            features: DataFrame or dict with feature values
            
        Returns:
            Predicted demand score
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call train_models() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Ensure correct feature order
        features = features[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.best_model.predict(features_scaled)
        
        return prediction[0]
    
    def _save_model(self):
        """Save best model and scaler to disk"""
        model_path = os.path.join(self.model_dir, 'best_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        metadata_path = os.path.join(self.model_dir, 'metadata.pkl')
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump({
            'model_name': self.best_model_name,
            'feature_names': self.feature_names
        }, metadata_path)
        
        print(f"✅ Model saved to {model_path}")
    
    def load_model(self):
        """Load saved model from disk"""
        model_path = os.path.join(self.model_dir, 'best_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        metadata_path = os.path.join(self.model_dir, 'metadata.pkl')
        
        if not os.path.exists(model_path):
            return False
        
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        metadata = joblib.load(metadata_path)
        self.best_model_name = metadata['model_name']
        self.feature_names = metadata['feature_names']
        
        print(f"✅ Model loaded: {self.best_model_name}")
        return True
    
    def get_model_comparison(self, results):
        """Generate model comparison DataFrame"""
        comparison_data = []
        
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Train MAE': result['train_mae'],
                'Test MAE': result['test_mae'],
                'Train RMSE': result['train_rmse'],
                'Test RMSE': result['test_rmse'],
                'Train R²': result['train_r2'],
                'Test R²': result['test_r2']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.round(4)
        
        return df
    
    def explain_metrics(self):
        """Explain evaluation metrics for non-technical users"""
        explanations = {
            'MAE': {
                'name': 'Mean Absolute Error',
                'meaning': 'Average prediction error in same units as target',
                'interpretation': 'Lower is better. Shows typical prediction mistake.',
                'example': 'MAE of 2.5 means predictions are off by 2.5 units on average'
            },
            'RMSE': {
                'name': 'Root Mean Squared Error',
                'meaning': 'Square root of average squared errors',
                'interpretation': 'Lower is better. Penalizes large errors more than MAE.',
                'example': 'RMSE of 3.0 means model has some larger prediction errors'
            },
            'R²': {
                'name': 'R-squared (Coefficient of Determination)',
                'meaning': 'Proportion of variance explained by model (0 to 1)',
                'interpretation': 'Higher is better. 0.8 = 80% of variance explained.',
                'example': 'R² of 0.85 means model captures 85% of demand patterns'
            }
        }
        
        return explanations

from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import traceback
from typing import Dict, Any
import logging
from datetime import datetime

# Import your ML pipeline modules
from api_client import ProductPriceAPI
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictionAPI:
    """
    REST API for product price prediction service.
    Production-ready Flask API integrating complete ML pipeline.
    
    Endpoints:
    - POST /predict: Single product price prediction
    - POST /batch_predict: Multiple products price prediction
    - GET /health: Health check
    - GET /models: Available models
    """

    def __init__(self):
        self.api_client = ProductPriceAPI()
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.loaded_model = None
        self.model_path = "best_price_model.pkl"

    def load_model(self) -> bool:
        """Load trained model for predictions."""
        try:
            self.loaded_model = self.model_trainer.load_model(self.model_path)
            logger.info(f"Model loaded successfully: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def predict_price(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict optimal price for single product.
        
        Args:
            product_data: Dictionary with product features
            
        Returns:
            Dictionary with prediction and recommendation
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([product_data])
            
            # Clean data
            df_clean = self.data_cleaner.clean_data(df)
            
            # Engineer features
            df_features = self.feature_engineer.engineer_features(df_clean)
            
            # Prepare for prediction
            X = df_features.drop(columns=['price', 'original_price'])
            
            # Handle categorical columns
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                X[col] = pd.factorize(X[col])[0]
            
            # Predict
            if self.loaded_model is not None:
                predicted_price = float(self.loaded_model.predict(X)[0])
                
                # Generate recommendation
                price_gap = predicted_price - float(product_data.get('price', 0))
                if price_gap < -10:
                    recommendation = "INCREASE_PRICE"
                elif price_gap > 10:
                    recommendation = "DECREASE_PRICE"
                else:
                    recommendation = "OPTIMAL_PRICE"
                
                return {
                    "success": True,
                    "predicted_price": round(predicted_price, 2),
                    "current_price": float(product_data.get('price', 0)),
                    "price_gap": round(price_gap, 2),
                    "recommendation": recommendation,
                    "confidence": 0.95  # Placeholder
                }
            else:
                return {"success": False, "error": "Model not loaded"}
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"success": False, "error": str(e)}

    def batch_predict(self, products_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch price prediction for multiple products.
        
        Args:
            products_data: List of product dictionaries
            
        Returns:
            Batch prediction results
        """
        try:
            results = []
            for product in products_data:
                result = self.predict_price(product)
                results.append(result)
            
            return {
                "success": True,
                "batch_size": len(products_data),
                "results": results,
                "summary": {
                    "avg_predicted_price": np.mean([r['predicted_price'] for r in results if r['success']]),
                    "recommendations": {
                        "increase": sum(1 for r in results if r.get('recommendation') == "INCREASE_PRICE"),
                        "decrease": sum(1 for r in results if r.get('recommendation') == "DECREASE_PRICE"),
                        "optimal": sum(1 for r in results if r.get('recommendation') == "OPTIMAL_PRICE")
                    }
                }
            }
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return {"success": False, "error": str(e)}

# Initialize API
price_api = PricePredictionAPI()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": price_api.loaded_model is not None
    })

# Model status endpoint
@app.route('/models', methods=['GET'])
def get_models():
    """Get available models status."""
    return jsonify({
        "model_path": price_api.model_path,
        "model_loaded": price_api.loaded_model is not None,
        "service_ready": price_api.loaded_model is not None
    })

# Single prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_price():
    """
    Single product price prediction endpoint.
    
    Request body:
    {
        "product_name": "iPhone 15 Pro",
        "brand": "Apple",
        "platform": "Amazon",
        "rating": 4.8,
        "review_count": 1250,
        "price": 999.99,
        "category": "smartphones"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        result = price_api.predict_price(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Predict endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Batch prediction endpoint
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch price prediction endpoint.
    
    Request body:
    [
        {...product1...},
        {...product2...},
        ...
    ]
    """
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({"error": "Expected list of products"}), 400
        
        result = price_api.batch_predict(data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Batch predict endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Train and deploy endpoint (admin)
@app.route('/train', methods=['POST'])
def train_model():
    """
    Train new model endpoint (admin access).
    
    Request body:
    {
        "category": "smartphones",
        "num_products": 500
    }
    """
    try:
        data = request.get_json()
        category = data.get('category', 'smartphones')
        num_products = data.get('num_products', 200)
        
        # Run complete pipeline
        api_client = ProductPriceAPI()
        trainer = ModelTrainer()
        
        # Fetch data
        raw_data = api_client.fetch_product_data(category, num_products)
        
        # Clean and engineer
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_data(raw_data)
        
        engineer = FeatureEngineer()
        engineered_data = engineer.engineer_features(cleaned_data)
        
        # Train models
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            engineered_data, 'price'
        )
        
        trainer.train_models(X_train, y_train)
        metrics = trainer.evaluate_models(X_test, y_test)
        
        # Save best model
        best_model_name = trainer.best_model
        trainer.save_model(best_model_name, price_api.model_path)
        
        price_api.load_model()
        
        return jsonify({
            "success": True,
            "best_model": best_model_name,
            "test_metrics": metrics[best_model_name],
            "total_models_trained": len(trainer.trained_models)
        })
        
    except Exception as e:
        logger.error(f"Train endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Root endpoint with docs
@app.route('/', methods=['GET'])
def home():
    """API documentation and Swagger-like interface."""
    return render_template('index.html')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Load model on startup
    if price_api.load_model():
        logger.info("🚀 Price Prediction API starting...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("❌ Failed to load model. Exiting.")

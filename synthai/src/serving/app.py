"""
Serving module for the SynthAI Model Training Framework.
This module provides a FastAPI application for serving models.
"""
import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
from synthai.src.models.model_factory import BaseModel as SynthAIModel
from synthai.src.data.preprocessor import DataPreprocessor

app = FastAPI(title="SynthAI Model Serving", description="API for serving SynthAI models")

# Global variables to hold model and preprocessor
model_wrapper = None
preprocessor = None
model_metadata = {}

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[Any]
    metadata: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Load model and preprocessor on startup."""
    global model_wrapper, preprocessor, model_metadata
    
    model_path = os.environ.get("SYNTHAI_MODEL_PATH")
    if not model_path:
        print("Warning: SYNTHAI_MODEL_PATH not set.")
        return

    try:
        # Load model
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict) and 'model' in data:
            model_wrapper = SynthAIModel()
            model_wrapper.model = data['model']
            model_wrapper.metadata = data.get('metadata', {})
            model_metadata = model_wrapper.metadata
        else:
            # Fallback for old format or raw model
            model_wrapper = SynthAIModel()
            model_wrapper.model = data
            
        # Load preprocessor if available (usually in same dir)
        model_dir = os.path.dirname(model_path)
        # Try to find a preprocessor file
        import glob
        preprocessor_files = glob.glob(os.path.join(model_dir, "preprocessor_*.pkl"))
        
        if preprocessor_files:
            # Use the most recent one
            preprocessor_path = sorted(preprocessor_files)[-1]
            with open(preprocessor_path, 'rb') as f:
                prep_data = pickle.load(f)
                
            preprocessor = DataPreprocessor(
                schema=prep_data['schema'],
                random_state=prep_data['random_state'],
                test_size=prep_data['test_size']
            )
            preprocessor.transformers = prep_data['transformers']
            print(f"Loaded preprocessor from {preprocessor_path}")
        else:
            print("Warning: No preprocessor found.")
            
        print(f"Loaded model from {model_path}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_type": model_metadata.get("model_type", "unknown")}

@app.get("/metadata")
def get_metadata():
    """Get model metadata."""
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make predictions."""
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(request.data)
        
        # Preprocess if we have a preprocessor
        if preprocessor:
            # We need to handle the preprocessing carefully
            # The preprocessor expects the full schema columns
            # For now, we'll assume the input matches the schema features
            
            # We need to replicate the preprocessing logic but for inference
            # This is a simplified version - ideally we'd refactor DataPreprocessor to have a transform method
            # that works on new data without target
            
            # For now, let's try to use the transformers directly
            X_processed_list = []
            
            for feature in preprocessor.schema["features"]:
                name = feature["name"]
                if name not in df.columns:
                    # If feature missing, fill with default? or error?
                    # For now, error
                    raise ValueError(f"Missing feature: {name}")
                
                col_data = df[name]
                
                # Apply transformers
                # Note: This relies on the naming convention in DataPreprocessor
                
                # Imputation
                if f"{name}_imputer" in preprocessor.transformers:
                    imputer = preprocessor.transformers[f"{name}_imputer"]
                    col_data_reshaped = col_data.values.reshape(-1, 1)
                    col_data_imputed = imputer.transform(col_data_reshaped)
                else:
                    col_data_imputed = col_data.values.reshape(-1, 1)
                
                # Scaling/Encoding
                if f"{name}_scaler" in preprocessor.transformers:
                    scaler = preprocessor.transformers[f"{name}_scaler"]
                    col_processed = scaler.transform(col_data_imputed)
                elif f"{name}_encoder" in preprocessor.transformers:
                    encoder = preprocessor.transformers[f"{name}_encoder"]
                    # Check if it's OneHot or Label
                    if isinstance(encoder, OneHotEncoder):
                        col_processed = encoder.transform(col_data_imputed)
                    else:
                        # LabelEncoder doesn't have transform for unseen labels usually, 
                        # and for inference we might need to handle this.
                        # But standard LabelEncoder in sklearn doesn't support transform well for new data
                        # if it wasn't seen. 
                        # For simplicity, we assume valid labels or use a custom safe transform
                        col_processed = encoder.transform(col_data_imputed.ravel()).reshape(-1, 1)
                elif f"{name}_vectorizer" in preprocessor.transformers:
                    vectorizer = preprocessor.transformers[f"{name}_vectorizer"]
                    col_processed = vectorizer.transform(col_data.fillna("")).toarray()
                else:
                    col_processed = col_data_imputed
                
                X_processed_list.append(col_processed)
            
            if len(X_processed_list) > 1:
                X = np.hstack(X_processed_list)
            else:
                X = X_processed_list[0]
                
        else:
            # No preprocessor, assume data is ready
            X = df.values
            
        # Predict
        predictions = model_wrapper.model.predict(X)
        
        return {"predictions": predictions.tolist(), "metadata": {"count": len(predictions)}}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """Start the serving server."""
    os.environ["SYNTHAI_MODEL_PATH"] = model_path
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    start_server(args.model_path, args.host, args.port)

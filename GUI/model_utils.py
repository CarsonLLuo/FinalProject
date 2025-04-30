import numpy as np
from test_model import create_model, preprocess_image

# Global variable to store model instance
_model = None

def get_model():
    
    global _model
    if _model is None:
        _model = create_model()
        weights_path = "/Users/carson/Desktop/code/GUI/vit_hybrid_spt_20250329_154643/best_model.h5"
        try:
            # First try normal loading
            _model.load_weights(weights_path)
        except:
            # If failed, try loading by name
            _model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return _model

def mock_prediction(image_array):
    """
    Makes a prediction using the loaded model on the preprocessed image array.
    
    Args:
        image_array (numpy.ndarray): Preprocessed image array with shape (1, height, width, channels)
        
    Returns:
        numpy.ndarray: Model prediction probabilities
    """
    model = get_model()
    prediction = model.predict(image_array, verbose=0)
    return prediction 
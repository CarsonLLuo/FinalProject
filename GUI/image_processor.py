import numpy as np
from PIL import Image

def preprocess_image(image):
    
    # Ensure image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize image to model input dimensions
    image = image.resize((160, 160))
    image_array = np.array(image)
    
    # Normalize
    image_array = image_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array 
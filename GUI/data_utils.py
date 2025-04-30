import pandas as pd
import numpy as np
from PIL import Image
from test_model import preprocess_image as model_preprocess_image

def get_sample_statistics():
    
    return {
        "Accuracy": "94.2%",
        "Sensitivity": "92.8%",
        "Specificity": "95.6%",
        "F1 Score": "93.7%"
    }

def get_cancer_types():
    
    data = {
        "Type": ["Ductal Carcinoma In Situ (DCIS)", "Invasive Ductal Carcinoma (IDC)", 
                "Invasive Lobular Carcinoma (ILC)", "Triple Negative Breast Cancer"],
        "Frequency": ["20-25%", "70-80%", "10-15%", "15-20%"],
        "Description": [
            "Abnormal cells in the milk duct, non-invasive", 
            "Cancer that begins in the ducts and invades surrounding tissue",
            "Cancer that begins in the lobules and invades surrounding tissue",
            "Lacks receptors for estrogen, progesterone, and HER2"
        ]
    }
    return pd.DataFrame(data)

def preprocess_image(image):
    
    if isinstance(image, str):
        return model_preprocess_image(image)
    else:
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            result = model_preprocess_image(tmp.name)
            os.unlink(tmp.name)  
            return result 
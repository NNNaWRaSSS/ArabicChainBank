#!/usr/bin/env python3
"""
Script to help diagnose and fix model loading issues with transformers library.
This addresses the ValueError when trying to load a model with incompatible configuration.
"""

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.models.auto.auto_factory import _get_model_class
import warnings

def diagnose_model(model_name_or_path, token=None):
    """
    Diagnose what type of model this is and suggest the correct loading method.
    
    Args:
        model_name_or_path (str): Model name or path
        token (str, optional): HuggingFace token for authentication
    """
    print(f"Diagnosing model: {model_name_or_path}")
    print("-" * 50)
    
    try:
        # Load the configuration first to understand the model type
        config = AutoConfig.from_pretrained(model_name_or_path, token=token, trust_remote_code=True)
        print(f"✓ Model configuration loaded successfully")
        print(f"  Configuration class: {config.__class__.__name__}")
        print(f"  Model type: {getattr(config, 'model_type', 'Unknown')}")
        
        # Check what auto classes support this config
        from transformers import (
            AutoModelForCausalLM, 
            AutoModelForMaskedLM, 
            AutoModelForSequenceClassification,
            AutoModelForImageClassification,
            AutoModelForObjectDetection,
            AutoModel
        )
        
        auto_classes = [
            ("AutoModelForCausalLM", AutoModelForCausalLM),
            ("AutoModelForMaskedLM", AutoModelForMaskedLM), 
            ("AutoModelForSequenceClassification", AutoModelForSequenceClassification),
            ("AutoModelForImageClassification", AutoModelForImageClassification),
            ("AutoModelForObjectDetection", AutoModelForObjectDetection),
            ("AutoModel", AutoModel),
        ]
        
        supported_classes = []
        for class_name, auto_class in auto_classes:
            try:
                if config.__class__ in auto_class._model_mapping.keys():
                    supported_classes.append(class_name)
            except AttributeError:
                continue
        
        if supported_classes:
            print(f"✓ Supported Auto classes: {', '.join(supported_classes)}")
        else:
            print("⚠ No standard Auto classes support this configuration")
            print("  This might be a custom model that requires special handling")
        
        return config, supported_classes
        
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return None, []

def load_model_correctly(model_name_or_path, token=None):
    """
    Attempt to load the model using the correct Auto class.
    
    Args:
        model_name_or_path (str): Model name or path
        token (str, optional): HuggingFace token for authentication
    """
    config, supported_classes = diagnose_model(model_name_or_path, token)
    
    if not config or not supported_classes:
        print("❌ Cannot determine how to load this model")
        return None, None
    
    print(f"\nAttempting to load model with {supported_classes[0]}...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            token=token, 
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded successfully")
        
        # Load model with the first supported auto class
        if "AutoModelForCausalLM" in supported_classes:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                token=token, 
                trust_remote_code=True
            )
        elif "AutoModelForMaskedLM" in supported_classes:
            from transformers import AutoModelForMaskedLM
            model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path, 
                token=token, 
                trust_remote_code=True
            )
        elif "AutoModel" in supported_classes:
            model = AutoModel.from_pretrained(
                model_name_or_path, 
                token=token, 
                trust_remote_code=True
            )
        else:
            # Try with the first supported class
            class_name = supported_classes[0]
            print(f"Using {class_name} for loading...")
            exec(f"from transformers import {class_name}")
            model_class = eval(class_name)
            model = model_class.from_pretrained(
                model_name_or_path, 
                token=token, 
                trust_remote_code=True
            )
        
        print("✓ Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

def alternative_loading_methods(model_name_or_path, token=None):
    """
    Show alternative loading methods if standard Auto classes fail.
    """
    print("\nAlternative loading methods:")
    print("-" * 30)
    
    # Method 1: Direct model loading with trust_remote_code
    print("1. If this is a custom model, try loading with trust_remote_code=True:")
    print(f"""
from transformers import AutoConfig, AutoModel, AutoTokenizer

config = AutoConfig.from_pretrained("{model_name_or_path}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{model_name_or_path}", trust_remote_code=True)
model = AutoModel.from_pretrained("{model_name_or_path}", trust_remote_code=True)
""")
    
    # Method 2: Check if it's a vision model
    print("2. If this is a vision model (NAT = Neighborhood Attention Transformer):")
    print(f"""
# NAT models are typically for computer vision, not text generation
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("{model_name_or_path}")
model = AutoModelForImageClassification.from_pretrained("{model_name_or_path}")
""")
    
    # Method 3: Manual class loading
    print("3. If you know the specific model class:")
    print(f"""
# Replace 'SpecificModelClass' with the actual class name
from transformers import SpecificModelClass, AutoTokenizer

model = SpecificModelClass.from_pretrained("{model_name_or_path}")
tokenizer = AutoTokenizer.from_pretrained("{model_name_or_path}")
""")

if __name__ == "__main__":
    # Example usage - replace with your actual model name
    # model_name = "your_model_name_here"
    # token = "your_hf_token_here"  # Optional
    
    print("Model Loading Diagnostics Tool")
    print("=" * 50)
    print("This script helps diagnose and fix model loading issues.")
    print("\nTo use this script:")
    print("1. Replace 'your_model_name_here' with your actual model name")
    print("2. Optionally provide your HuggingFace token")
    print("3. Run the diagnostic functions")
    
    # Uncomment and modify these lines to test with your model:
    # model, tokenizer = load_model_correctly(model_name, token)
    # if model is None:
    #     alternative_loading_methods(model_name, token)
import torch
from ultralytics import YOLO
import os


# GPU & TENSOR CORE OPTIMIZATION

def main():
    # Enable TF32 for RTX 3060 Ti Tensor Cores
    torch.set_float32_matmul_precision('high')
    
    # Benchmark mode for speed optimization
    torch.backends.cudnn.benchmark = True  

    print("--- GPU Status ---")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("------------------\n")

    
    # PATHS
    
    # data.yaml is in the same folder as this script
    DATA_YAML = "data.yaml"      
    MODEL_SAVE_DIR = "runs"        
    MODEL_NAME = "neuralens"   

    
    # LOAD MODEL
    
    model = YOLO("runs/neuralens/weights/last.pt")

    
    # TRAINING CONFIGURATION
    
    # Ensure save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    model.train(
        resume=True,
        data=DATA_YAML,
        epochs=100,           
        imgsz=640,            
        
        # SPEED & HARDWARE SETTINGS
        device=0,             # RTX 3060 Ti
        amp=True,             # Tensor Core Mixed Precision
        batch=32,             # Optimized for 8GB VRAM
        workers=8,            
        
        # HYPERPARAMETERS
        optimizer="AdamW",
        lr0=0.001,
        patience=20,          
        
        # OUTPUT SETTINGS
        project=MODEL_SAVE_DIR,
        name=MODEL_NAME,
        exist_ok=True,        
        plots=True,           
        verbose=True,

        # DATA AUGMENTATION
        mosaic=1.0,           
        fliplr=0.5,           
        hsv_h=0.015,          
        hsv_s=0.7,
        hsv_v=0.4
    )

if __name__ == "__main__":
    main()
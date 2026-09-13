from print_config import print_config
import config as cfg
import os
import sys
import torch
from config import NUM_CLASSES,MODEL_TIMM,DEVICE,MODEL_FILE_PATH,MODEL_NAME,OPTIMIZER_LR
from timm import create_model
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from model_manager import ModelManager
from evaluate import evaluate
import numpy as np

if __name__ == "__main__":
    from dataloader import test_loader,val_loader,train_loader
    print_config(cfg, title="Training Configuration")
    torch.backends.cudnn.benchmark = True
    model = create_model(MODEL_TIMM, pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER_LR)
    manager = ModelManager(model,optimizer)
    manager.criterion.to(DEVICE)
    torch.cuda.empty_cache()
    manager.train(train_loader, val_loader)
    torch.cuda.empty_cache()
    results, y_true, y_pred, y_prob, chosen_thresholds = evaluate(model,test_loader, save_file_path=os.path.join(os.path.join(MODEL_FILE_PATH, MODEL_NAME), "test_result5.csv"),printr=True)
# config.py
import os
import torch
import torch.nn as nn
MODEL_TIMM = 'resnet50'
MODEL_NAME = "0708_EfficientNet"
DATASET_NAME = "0303_original"

DISEASE_LABELS = [
    "Atelectasis",
    "Hernia",
    "Cardiomegaly",
    "Infiltration",
    "Consolidation",
    "Mass",
    "Edema",
    "Nodule",
    "Effusion",
    "Pleural_Thickening",
    "Emphysema",
    "Pneumonia",
    "Fibrosis",
    "Pneumothorax",
    "No Finding"
]

NUM_CLASSES = 15
FILE_PATH = "C:/CheSwinV3"
CODE_FILE_PATH = os.path.join(FILE_PATH, "Code/exp_original")
MODEL_FILE_PATH = os.path.join(FILE_PATH, "ModelResult")
DATASET_FILE_PATH = os.path.join(FILE_PATH, "DataSet")
IMAGE_SIZE = 384
IMAGE_ROOT = "C:/NIH_ChestXray14"
dataset_full_path = os.path.normpath(os.path.join(DATASET_FILE_PATH, DATASET_NAME))
CSV_PATHS = {
    'train': os.path.normpath(os.path.join(dataset_full_path, "train.csv")),
    'val':   os.path.normpath(os.path.join(dataset_full_path, "val.csv")),
    'test':  os.path.normpath(os.path.join(dataset_full_path, "test.csv")),
}

NUM_WORKERS = 15
BATCH_SIZE = 8
NUM_EPOCHS = 10
OPTIMIZER_LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.BCEWithLogitsLoss()
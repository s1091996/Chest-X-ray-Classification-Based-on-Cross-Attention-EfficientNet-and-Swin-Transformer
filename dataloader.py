import pandas as pd
from torch.utils.data import  DataLoader

from dataset import NIHDataset,val_test_transform,train_transform
from config import LABEL_COLS, CSV_PATHS, IMAGE_ROOT, BATCH_SIZE, NUM_WORKERS

def get_dataloaders_from_split():
    
    train_df = pd.read_csv(CSV_PATHS['train'])
    val_df = pd.read_csv(CSV_PATHS['val'])
    test_df = pd.read_csv(CSV_PATHS['test'])
    
    train_dataset = NIHDataset(train_df, IMAGE_ROOT, LABEL_COLS,train_transform)
    val_dataset = NIHDataset(val_df, IMAGE_ROOT, LABEL_COLS,val_test_transform)
    test_dataset = NIHDataset(test_df, IMAGE_ROOT, LABEL_COLS,val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_dataloaders_from_split()
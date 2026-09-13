import torch
import shutil
import os
from tqdm import tqdm
from config import NUM_EPOCHS, DEVICE, CRITERION,MODEL_FILE_PATH, CODE_FILE_PATH,MODEL_NAME,ACCUMULATION_STEPS,EARLY_STOPPING,MODEL_FILE_PATH
from evaluate import evaluate
from torch import amp
import os

class ModelManager:
    def __init__(self, model, optimizer):
        """
        初始化 ModelManager，建立資料夾並備份 config.py
        """
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = CRITERION
        self.device = DEVICE
        self.model_name = MODEL_NAME
        self.epoch = 0
        self.loss = None
        self.accumulation_steps = ACCUMULATION_STEPS
        self.best_f1 = -float("inf")
        self.best_epoch = 0
        self.f1_thresholds = None
        self.epochs_without_improvement = 0
        self.early_stopping_patience = EARLY_STOPPING

        self.scaler = amp.GradScaler(
            "cuda", 
            enabled=(self.device.type == "cuda")
        )

        full_save_path = os.path.join(MODEL_FILE_PATH, MODEL_NAME)
        
        if not os.path.exists(full_save_path):
            os.makedirs(full_save_path)
            print(f"[ModelManager] 建立新資料夾: {full_save_path}")
        else:
            print(f"[ModelManager] 使用現有資料夾: {full_save_path}")

        self.model_path = full_save_path

        self._backup_config()

    def _backup_config(self):
        src_config_path = os.path.join(CODE_FILE_PATH, "config.py")
        dst_config_path = os.path.join(self.model_path, "config.py")

        try:
            if not os.path.exists(src_config_path):
                print(f"[Warning] 在 {src_config_path} 找不到 config.py，無法備份！")
                return
            shutil.copy(src_config_path, dst_config_path)
            print(f"[ModelManager] Config 已備份至: {dst_config_path}")

        except Exception as e:
            print(f"[Error] 備份 config.py 時發生錯誤: {e}")

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc="Training", leave=False)
        self.optimizer.zero_grad()

        for i, (imgs, labels) in enumerate(loop):
            imgs = imgs.to(self.device)
            labels = labels.float().to(self.device)

            with amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                
            loss = loss / self.accumulation_steps
            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(train_loader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad() 

            total_loss += loss.item() * self.accumulation_steps
            loop.set_postfix(loss=loss.item() * self.accumulation_steps)

        avg_loss = total_loss / len(train_loader)
        self.loss = avg_loss

        return avg_loss

    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS):
        auc_history = []
        f1_history = []

        for epoch in range(self.epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss = self.train_one_epoch(train_loader)
            self.epoch = epoch + 1

            save_csv_path = os.path.join(self.model_path, f"{self.model_name}_epoch{self.epoch}_valresult.csv")
            results, _, _, _, chosen_thresholds = evaluate(model=self.model, loader=val_loader,threshold=None, save_file_path=save_csv_path)
            self.f1_thresholds = chosen_thresholds
            macro_acc = results["macro_avg"]["acc"]
            macro_f1 = results["macro_avg"]["f1"]
            macro_auc = results["macro_avg"]["auc"]

            print(f"[{self.model_name} epoch {self.epoch}] Train Loss: {train_loss:.4f} | Val Acc: {macro_acc:.4f} | Val F1: {macro_f1:.4f} | Val AUC: {macro_auc:.4f}")

            auc_history.append(macro_auc)
            f1_history.append(macro_f1)

            if macro_f1 > self.best_f1:
                self.best_f1 = macro_f1
                self.best_epoch = self.epoch
                self.epochs_without_improvement = 0

            else:
                self.epochs_without_improvement += 1
                print(f"F1 沒有提升 ({self.epochs_without_improvement}/3)")

            if self.model_name:
                self.save_statement()

            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly Stopping：連續 3 個 Epoch 沒有提升")
                print(f"最佳 F1-score: {self.best_f1:.4f}")
                print(f"最佳 Epoch: {self.best_epoch}")
                break
            
        print(f"\n訓練結束，最佳 F1-score: {self.best_f1:.4f}，最佳 Epoch: {self.best_epoch}")
        self.load_statement(self.best_epoch)
        result_path = os.path.join(os.path.join(MODEL_FILE_PATH, MODEL_NAME), "BestModel.csv")
        evaluate(
            model=self.model,
            loader=val_loader,
            threshold=self.f1_thresholds,
            save_file_path=result_path,
            printr=True
        )  

    def save_statement(self):
        checkpoint_file_name = f"{self.model_name}_epoch{self.epoch}_checkpoint.pth"
        checkpoint_save_path = os.path.join(self.model_path, checkpoint_file_name)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': self.loss,
            'f1_thresholds': self.f1_thresholds
        }
        torch.save(checkpoint, checkpoint_save_path)
        print(f"Checkpoint saved to {checkpoint_save_path}")

    def load_statement(self, epoch,file_path=None):
        """
        輸入 Epoch 數字即可讀取對應的模型權重
        Example: manager.load_statement(5)
        """
        if file_path:
            checkpoint_path = file_path
        else :
            checkpoint_file_name = f"{self.model_name}_epoch{epoch}_checkpoint.pth"
            checkpoint_path = os.path.join(self.model_path, checkpoint_file_name)

        # 3. 檢查檔案是否存在
        if not os.path.exists(checkpoint_path):
            print(f"❌ [Error] 找不到 Checkpoint 檔案: {checkpoint_path}")
            print(f"   請確認 Epoch {epoch} 是否已經訓練並存檔。")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            if hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.epoch = checkpoint['epoch']
            self.loss = checkpoint.get('loss', None)
            self.f1_thresholds = checkpoint.get('f1_thresholds', None)

            del checkpoint

            print(f"✅ [Success] 成功載入 Epoch {self.epoch} 模型權重")
            print(f"   路徑: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ [Error] 讀取 Checkpoint 時發生錯誤: {e}")
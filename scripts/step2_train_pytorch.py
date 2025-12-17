"""
Step 2: PyTorch EfficientNetV2 Training (Final Weighted Version)
수정사항:
1. EfficientNetV2-S 모델 적용 (Step3와 통일)
2. Class Weight 자동 계산 -> OTE(소수 클래스) 학습 강화
3. 5 Epoch 이후 Backbone Unfreeze (미세 조정 학습)

실행: python step2_train_pytorch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Config:
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    DATASET_DIR = PROJECT_ROOT / "dataset_full"
    MODELS_DIR = PROJECT_ROOT / "models_full"
    
    BATCH_SIZE = 16  
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 0  
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def imread_safe(path):
    try:
        with open(path, 'rb') as f: arr = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except: return None

class DISEDataset(Dataset):
    def __init__(self, csv_path, dataset_root):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = Path(dataset_root)
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.dataset_root / row['image_path']
        img = imread_safe(img_path)
        
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            # 간단한 Augmentation 
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
        else:
            img = np.zeros((224, 224, 3))
        
        img = torch.FloatTensor(img).permute(2, 0, 1)
        phase = torch.FloatTensor([row['phase_input']])
        label = torch.LongTensor([row['cause_target']])[0]
        return img, phase, label

class MultiInputEfficientNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Step3와 동일한 모델 아키텍처 사용
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.img_fc = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3))
        self.phase_fc = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(288, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
    
    def forward(self, img, phase):
        x1 = self.img_fc(self.backbone(img))
        x2 = self.phase_fc(phase)
        return self.classifier(torch.cat([x1, x2], dim=1))
    
    def unfreeze_backbone(self):
        print(" Unfreezing backbone for fine-tuning...")
        for param in self.backbone.parameters():
            param.requires_grad = True

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        print(f" Device: {self.device}")

    def calculate_weights(self, df):
        labels = df['cause_target'].values
        classes = np.unique(labels)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        
        weight_tensor = torch.zeros(5).to(self.device)
        # 매핑이 비어있는 클래스 방지
        for cls, w in zip(classes, weights):
            weight_tensor[cls] = w
            
        print("\n[Class Weights Applied]")
        labels_map = {0:'no', 1:'Velum', 2:'Oropharynx', 3:'Tongue', 4:'Epiglottis'}
        counts = df['cause_target'].value_counts().sort_index()
        for i in range(5):
            cnt = counts.get(i, 0)
            w = weight_tensor[i].item()
            print(f"  Class {i} ({labels_map.get(i)}): {cnt} samples -> Weight: {w:.2f}")
        return weight_tensor
    
    def train(self):
        train_csv = self.config.DATASET_DIR / 'train.csv'
        if not train_csv.exists():
            print(" Train.csv not found!")
            return

        train_df = pd.read_csv(train_csv)
        train_dataset = DISEDataset(train_csv, self.config.DATASET_DIR)
        val_dataset = DISEDataset(self.config.DATASET_DIR / 'val.csv', self.config.DATASET_DIR)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=self.config.NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS)
        
        # 가중치 계산
        class_weights = self.calculate_weights(train_df)
        
        model = MultiInputEfficientNet(num_classes=5).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        self.config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        best_acc = 0
        
        print(f"\nTraining Started ({self.config.EPOCHS} epochs)...")
        
        for epoch in range(self.config.EPOCHS):
            # 5 Epoch 후 백본 풀기
            if epoch == 5:
                model.unfreeze_backbone()
                
            model.train()
            train_loss, correct, total = 0, 0, 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for imgs, phases, labels in pbar:
                imgs, phases, labels = imgs.to(self.device), phases.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(imgs, phases)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            train_acc = 100. * correct / total
            
            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for imgs, phases, labels in val_loader:
                    imgs, phases, labels = imgs.to(self.device), phases.to(self.device), labels.to(self.device)
                    outputs = model(imgs, phases)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            scheduler.step(val_acc)
            
            print(f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), self.config.MODELS_DIR / f'best_model.pth')
                print(f"  ★ Best Model Saved")
        
        torch.save(model.state_dict(), self.config.MODELS_DIR / 'final_model.pth')
        print("\n Training Complete!")

if __name__ == "__main__":
    Trainer(Config()).train()
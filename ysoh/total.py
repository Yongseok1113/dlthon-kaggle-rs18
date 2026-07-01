import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Dict

import clip
import imagehash

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_SIZE = 224


def get_transforms(img_size: int = IMG_SIZE):
    """
    Train 및 Validation을 위한 Albumentations 파이프라인
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # [수정됨] size 인자를 (height, width) 튜플 형태로 전달
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=0.5),        
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def create_dataloaders(df_cleaned: pd.DataFrame, img_dir: str, batch_size: int = 32):
    """
    정제된 DataFrame을 바탕으로 Train, Validation DataLoader를 생성합니다.
    """
    # 1. 클래스 비율을 유지하며 Train / Validation 분할 (8:2)
    # 캐글 특성상 5-Fold를 권장하지만, 베이스라인 구축을 위해 우선 Hold-out으로 구현합니다.
    df_train, df_val = train_test_split(
        df_cleaned, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_cleaned['label'] # 라벨 불균형 방지
    )
    
    print(f"[Log] Train 데이터: {len(df_train)}장 | Validation 데이터: {len(df_val)}장")
    
    # 2. Transform 로드
    train_transform, val_transform = get_transforms(img_size=IMG_SIZE)
    
    # 3. Dataset 인스턴스화
    train_dataset = SolarPanelDataset(df_train, img_dir, transform=train_transform)
    val_dataset = SolarPanelDataset(df_val, img_dir, transform=val_transform)
    
    # 4. DataLoader 인스턴스화
    # num_workers는 환경에 맞게 조절 (보통 CPU 코어 수의 2배 또는 4 사용)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, # GPU 전송 속도 향상
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader

class SolarPanelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        """
        :param df: 정제가 완료된 데이터프레임 (image_id, label 포함)
        :param img_dir: 이미지 파일들이 위치한 폴더 경로
        :param transform: Albumentations 증강 파이프라인
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id']
        label = row['label']
        
        img_path = os.path.join(self.img_dir, img_name)
        
        # OpenCV로 이미지 로드 (Albumentations는 BGR -> RGB 변환 필요)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 증강 적용
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # 이진 분류이므로 label을 float32 타입의 Tensor로 반환 (BCEWithLogitsLoss 사용을 위함)
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label

class SolarTestDataset(Dataset):
    def __init__(self, img_dir, img_ids, transform=None):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, img_name

class DatasetCleaner:
    def __init__(self, train_dir: str, test_dir: str, label_path: str, preprocessing_modules: List[str], ext: str = "jpg"):
        """
        태양광 패널 이미지 데이터셋 정제 및 필터링 클래스 (실제 패키지 적용 버전)
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.label_path = label_path
        self.preprocessing_modules = preprocessing_modules
        self.file_ext = "jpg"
        
        if os.path.exists(label_path):
            self.df_label = pd.read_csv(label_path)
            self.df_label.columns = ['image_id', 'label']
        else:
            raise FileNotFoundError(f"라벨 파일을 찾을 수 없습니다: {label_path}")
            
        self.df_cleaned = self.df_label.copy()
        self._embeddings_cache = {}

    def _extract_features(self) -> Dict[str, dict]:
        """
        CLIP 임베딩(OOD용)과 pHash(중복탐지용)를 한 번에 추출하여 캐싱합니다.
        """
        if not self._embeddings_cache:
            print("[Log] CLIP 모델 로드 및 특징/해시 추출 시작... (시간이 소요될 수 있습니다)")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            for img_id in self.df_label['image_id']:
                img_path = os.path.join(self.train_dir, img_id) + '.' + self.file_ext
                try:
                    # OpenCV 대신 PIL을 사용하여 CLIP 및 ImageHash 호환성 확보
                    pil_img = Image.open(img_path).convert("RGB")
                    
                    # 1. pHash 추출
                    phash_val = imagehash.phash(pil_img)
                    
                    # 2. CLIP 임베딩 추출
                    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model.encode_image(img_tensor).cpu().numpy().flatten()
                        
                    self._embeddings_cache[img_id] = {
                        'phash': phash_val,
                        'embedding': emb
                    }
                except Exception as e:
                    print(f"[Error] 이미지 로드 실패 ({img_id}): {e}")
                    
        return self._embeddings_cache

    def filtering_duplicate(self, hash_threshold: int = 5) -> pd.DataFrame:
        """
        pHash 기반 해밍 거리(Hamming Distance)를 계산하여 중복/유사 이미지를 탐지합니다.
        :param hash_threshold: 이 수치 이하의 거리를 가지면 중복으로 판단 (보통 5 이하)
        """
        print("[Log] 중복 이미지 탐지 시작 (filtering_duplicate)...")
        features = self._extract_features()
        img_ids = list(features.keys())
        
        flagged_records = []
        checked = set()
        
        for i in range(len(img_ids)):
            id_a = img_ids[i]
            if id_a in checked:
                continue
                
            hash_a = features[id_a]['phash']
            
            for j in range(i + 1, len(img_ids)):
                id_b = img_ids[j]
                if id_b in checked:
                    continue
                    
                hash_b = features[id_b]['phash']
                # 해시 간의 차이(해밍 거리)가 작을수록 유사한 이미지
                distance = hash_a - hash_b 
                
                if distance <= hash_threshold:
                    # 의심 점수: 거리가 0이면 1.0(완벽동일), 거리가 커질수록 점수 감소
                    suspicion_score = 1.0 - (distance / (hash_threshold + 1))
                    flagged_records.append({
                        'image_id': id_b, # B를 중복본으로 간주하여 제거 리스트에 추가
                        'suspicion_score': suspicion_score
                    })
                    checked.add(id_b)
                    
        df_res = pd.DataFrame(flagged_records, columns=['image_id', 'suspicion_score'])
        return df_res.sort_values(by='suspicion_score', ascending=False)

    def filtering_ood(self, contamination: float = 0.05) -> pd.DataFrame:
        """
        CLIP 임베딩과 scikit-learn IsolationForest를 이용해 OOD 이미지를 탐지합니다.
        """
        print("[Log] OOD(상관없는 이미지) 탐지 시작 (filtering_ood)...")
        features = self._extract_features()
        
        img_ids = list(features.keys())
        embeddings = np.array([features[uid]['embedding'] for uid in img_ids])
        
        clf = IsolationForest(contamination=contamination, random_state=42)
        preds = clf.fit_predict(embeddings) 
        scores = -clf.score_samples(embeddings) # 이상치일수록 양의 방향으로 높은 점수
        
        flagged_records = []
        for img_id, score, pred in zip(img_ids, scores, preds):
            if pred == -1: 
                flagged_records.append({
                    'image_id': img_id,
                    'suspicion_score': float(score)
                })
                
        df_res = pd.DataFrame(flagged_records, columns=['image_id', 'suspicion_score'])
        return df_res.sort_values(by='suspicion_score', ascending=False)

    def filtering_mislabel(self, oof_predictions: Dict[str, float] = None) -> pd.DataFrame:
        """
        OOF 예측값을 기반으로 오라벨링을 탐지합니다.
        (현재 Phase 1에서는 모델 학습 전이므로 외부 주입 혹은 Pass 형태입니다)
        """
        print("[Log] 오라벨링 이미지 탐지 시작 (filtering_mislabel)...")
        flagged_records = []
        
        if not oof_predictions:
            print("[Warning] oof_predictions이 제공되지 않아 오라벨링 필터링을 건너뜁니다.")
            return pd.DataFrame(columns=['image_id', 'suspicion_score'])

        for idx, row in self.df_label.iterrows():
            img_id = row['image_id']
            true_label = row['label']
            
            if img_id in oof_predictions:
                pred_prob_1 = oof_predictions[img_id]
                suspicion_score = abs(true_label - pred_prob_1)
                
                # 예측과 실제 라벨의 차이가 0.7 이상일 때 강한 의심
                if suspicion_score > 0.7:
                    flagged_records.append({
                        'image_id': img_id,
                        'suspicion_score': suspicion_score
                    })
                    
        df_res = pd.DataFrame(flagged_records, columns=['image_id', 'suspicion_score'])
        return df_res.sort_values(by='suspicion_score', ascending=False)

    def preprocessing(self, **kwargs) -> pd.DataFrame:
        print(f"\n[Pipeline] 전처리 및 필터링 파이프라인 구동 시작: {self.preprocessing_modules}")
        
        all_filtered_ids = set()
        summary_report = {}
        
        if 'duplicate' in self.preprocessing_modules:
            df_dup = self.filtering_duplicate(hash_threshold=kwargs.get('hash_threshold', 5))
            all_filtered_ids.update(df_dup['image_id'].tolist())
            summary_report['duplicate'] = len(df_dup)
            
        if 'ood' in self.preprocessing_modules:
            df_ood = self.filtering_ood(contamination=kwargs.get('ood_contamination', 0.05))
            all_filtered_ids.update(df_ood['image_id'].tolist())
            summary_report['ood'] = len(df_ood)
            
        if 'mislabel' in self.preprocessing_modules:
            df_mis = self.filtering_mislabel(oof_predictions=kwargs.get('oof_predictions', None))
            all_filtered_ids.update(df_mis['image_id'].tolist())
            summary_report['mislabel'] = len(df_mis)

        self.df_cleaned = self.df_label[~self.df_label['image_id'].isin(all_filtered_ids)].reset_index(drop=True)
        
        print("\n" + "="*40)
        print("[Pipeline Report] 필터링 요약")
        for module, count in summary_report.items():
            print(f" - {module.upper()} 모듈 제거 개수: {count}장")
        print(f" - 원본 데이터: {len(self.df_label)}장 -> 정제 후: {len(self.df_cleaned)}장")
        print("="*40 + "\n")
        
        return self.df_cleaned
    
    def remove_by_scores(self, 
                         score_dfs: Dict[str, pd.DataFrame], 
                         thresholds: Dict[str, float]) -> pd.DataFrame:
        """
        Phase 2: Phase 1에서 계산된 노이즈 의심 점수를 바탕으로 데이터를 최종 제거합니다.
        
        :param score_dfs: {'duplicate': df_dup, 'ood': df_ood, 'mislabel': df_mis} 형태의 딕셔너리
        :param thresholds: 각 노이즈 유형별로 제거를 결정할 점수 하한선 (이 점수 이상이면 제거)
        :return: 노이즈가 제거된 최종 Cleaned DataFrame
        """
        print("\n[Log] 사용자 지정 임계치 기반 데이터 제거 (Phase 2) 시작...")
        
        all_filtered_ids = set()
        removal_summary = {}

        for noise_type, df_score in score_dfs.items():
            if df_score is None or df_score.empty:
                removal_summary[noise_type] = 0
                continue
                
            # 해당 노이즈 유형의 임계치 가져오기 (지정되지 않았다면 기본값 0.0으로 설정하여 모두 제거 방지)
            threshold = thresholds.get(noise_type, float('inf'))
            
            # 임계치 이상인(즉, 노이즈일 확률이 매우 높은) 데이터만 필터링
            target_to_remove = df_score[df_score['suspicion_score'] >= threshold]
            ids_to_remove = target_to_remove['image_id'].tolist()
            
            all_filtered_ids.update(ids_to_remove)
            removal_summary[noise_type] = len(ids_to_remove)
            
            print(f" - [{noise_type.upper()}] Threshold >= {threshold:.3f} 적용: {len(ids_to_remove)}장 제거 대상 선정")

        # 원본 라벨에서 필터링 대상 ID들을 제외하여 최종 정제된 데이터프레임 생성
        self.df_cleaned = self.df_label[~self.df_label['image_id'].isin(all_filtered_ids)].reset_index(drop=True)

        print("\n" + "="*40)
        print("[Phase 2 Report] 점수 기반 제거 완료")
        for noise_type, count in removal_summary.items():
            print(f" - {noise_type.upper()} 제거: {count}장")
        print(f" - 총 고유 제거 장수 (중복 포함): {len(all_filtered_ids)}장")
        print(f" - 최종 잔존 데이터: {len(self.df_label)}장 -> {len(self.df_cleaned)}장")
        print("="*40 + "\n")

        return self.df_cleaned

def kaggle_submit(
        competition="rs-18-track-a", 
        submit_file_path="./submission_a.csv",
        message="default"
):
    from submit import submit, show_submission_status
    
    submit(competition, submit_file_path, message)
    show_submission_status(competition)


def run_filtering_pipeline(base_dir: str):
    """
    [모듈 1] 데이터 정제 및 필터링 파이프라인
    - 원본 데이터를 읽어 노이즈를 제거하고 결과를 CSV로 저장합니다.
    """
    print("\n--- [Module 1] Filtering Pipeline Start ---")
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test') 
    label_path = os.path.join(base_dir, 'train_labels.csv')
    cleaned_label_path = os.path.join(base_dir, 'train_label_cleaned.csv')

    cleaner = DatasetCleaner(
        train_dir=train_dir,
        test_dir=test_dir, 
        label_path=label_path, 
        preprocessing_modules=['duplicate', 'ood', 'mislabel']
    )
    
    df_cleaned_final = cleaner.preprocessing(
        hash_threshold=4, 
        ood_contamination=0.03
    )
    
    df_cleaned_final.to_csv(cleaned_label_path, index=False)
    print(f"정제 완료 및 저장됨: {cleaned_label_path}")


def run_training_pipeline(base_dir: str):
    """
    [모듈 2] 학습 데이터 로더 생성 파이프라인
    - 이미 정제된 CSV 파일을 불러와 즉시 학습 준비를 마칩니다.
    """
    print("\n--- [Module 2] Training Pipeline Start ---")
    train_dir = os.path.join(base_dir, 'train')
    cleaned_label_path = os.path.join(base_dir, 'train_label_cleaned.csv')

    if not os.path.exists(cleaned_label_path):
        raise FileNotFoundError(f"정제된 라벨 파일이 없습니다. 먼저 Filtering Pipeline을 실행하세요: {cleaned_label_path}")

    # 기존 정제된 CSV 명시적 로드
    print(f"정제된 파일 로드: {cleaned_label_path}")
    df_cleaned_final = pd.read_csv(cleaned_label_path)

    # DataLoader 생성
    train_loader, val_loader = create_dataloaders(
        df_cleaned=df_cleaned_final, 
        img_dir=train_dir, 
        batch_size=64
    )
    
    # 로더 테스트
    for images, labels in train_loader:
        print(f"Batch Image Shape: {images.shape}") 
        print(f"Batch Label Shape: {labels.shape}") 
        break

    run_experiment(train_loader, val_loader)

# ==========================================
# 1. 모델 정의 (EfficientNet Baseline)
# ==========================================
class SolarClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(SolarClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        # BCEWithLogitsLoss를 사용할 것이므로 sigmoid를 적용하지 않습니다.
        return self.model(x).squeeze(1)

# ==========================================
# 2. AUROC 중심의 학습 및 검증 루프
# ==========================================
def train_model(model, train_loader, val_loader, device, num_epochs=10):
    # BCEWithLogitsLoss: 모델 출력값(Logit)을 받아 내부적으로 Sigmoid를 처리
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_auc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        # ... (Train 루프 동일: loss = criterion(outputs, labels))
        
        model.eval()
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # 평가지표용 확률값 계산
                probs = torch.sigmoid(outputs) 
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # AUROC 계산
        val_auc = roc_auc_score(all_targets, all_probs)
        print(f"Epoch {epoch+1} | Val ROC-AUC: {val_auc:.5f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_weights = model.state_dict().copy()
            
    model.load_state_dict(best_model_weights)
    return model

# ==========================================
# 3. 실행 블록
# ==========================================
def run_experiment(train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[System] Using Device: {device}")

    model = SolarClassifier(model_name='efficientnet_b0', pretrained=True)
    model = model.to(device)

    # 학습 구동
    best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=10
    )
    
    # Best 모델 저장
    save_path = os.path.join(os.getcwd(), 'datasets', 'rs-18-track-a', 'efficientnet_b0_best_auc.pth')
    torch.save(best_model.state_dict(), save_path)
    print(f"\n[Save] 모델 저장 완료: {save_path}")


def generate_submission(model, test_dir, sample_sub_path, save_path, batch_size=64, img_size=IMG_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df_sub = pd.read_csv(sample_sub_path)
    test_ids = df_sub['id'].tolist()

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Dataset 클래스는 기존 SolarTestDataset 사용
    test_dataset = SolarTestDataset(test_dir, test_ids, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            # 0~1 사이의 확률값 추출
            probs = torch.sigmoid(logits) 
            predictions.extend(probs.cpu().numpy())
            
    df_sub['dusty_prob'] = predictions
    df_sub.to_csv(save_path, index=False)
    print(f"제출 파일 저장 완료: {save_path}")

if __name__ == "__main__":
    # 공통 베이스 경로
    BASE_DIR = os.path.join(os.getcwd(), 'datasets', 'rs-18-track-a')

    # ----------------------------------------------------------------
    # [실행 제어부] 
    # 필요에 따라 주석 처리하여 각 모듈을 독립적으로 실행합니다.
    # 추후 스크립트를 분리할 때는 각 함수만 따로 가져가면 됩니다.
    # ----------------------------------------------------------------

    # 1. 필터링 (필요시)
    # run_filtering_pipeline(BASE_DIR)
    
    # 2. 학습 (train_loader/val_loader를 반환받아 사용)
    model, train_loader, val_loader = run_training_pipeline(BASE_DIR)
    
    # 3. 추론 (학습된 가중치 로드)
    best_model = SolarClassifier()
    best_model.load_state_dict(torch.load('efficientnet_b0_best_auc.pth'))
    generate_submission(best_model, os.path.join(BASE_DIR, 'test'), ...)
    
    # summit
    # kaggle_submit(
    #     competition="rs-18-track-a", 
    #     submit_file_path="./submission_a.csv",
    #     message="default"
    # )
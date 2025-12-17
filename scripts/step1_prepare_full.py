"""
Step 1: Full Dataset Preparation (Fixed Mapping)
수정사항:
- 'Tongue_Base' (대문자 B) 매핑 추가 -> 데이터 누락 해결
- 그 외 혹시 모를 오타 방지용 매핑 추가

실행: python step1_prepare_full.py
"""

import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


def imread_safe(path):
    """한글 경로 안전 읽기"""
    try:
        with open(path, 'rb') as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None


def imwrite_safe(path, img):
    """한글 경로 안전 쓰기"""
    try:
        result, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if result:
            with open(path, 'wb') as f:
                f.write(buf.tobytes())
            return True
        return False
    except:
        return False


class Config:
    """설정"""
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    VELUM_DIR = PROJECT_ROOT / "Velum"
    OTE_DIR = PROJECT_ROOT / "OTE"
    
    DATASET_DIR = PROJECT_ROOT / "dataset_full"
    
    # 전체 데이터 사용
    SAMPLE_SIZE = None
    FRAMES_PER_VIDEO = 10
    IMAGE_SIZE = 224
    
    # 라벨 매핑 (수정됨: 대소문자 변형 모두 포함)
    PHASE_MAP = {'Velum': 0, 'OTE': 1}
    
    CAUSE_MAP = {
        'no': 0,
        
        # Velum
        'Velum': 1,
        
        # Oropharynx (구인두)
        'Oropharynx': 2,
        'Oropharynx_posterior_lateral_walls': 2,
        
        # Tongue (혀/설근) - 여기가 문제였음!
        'Tongue': 3,
        'Tongue_base': 3,  # 소문자 b
        'Tongue_Base': 3,  # [NEW] 대문자 B (범인 검거)
        'Tongue base': 3,  # [NEW] 띄어쓰기 버전 (혹시 몰라서 추가)
        
        # Epiglottis (후두개)
        'Epiglottis': 4
    }


class DataPreparation:
    """데이터 준비"""
    
    def __init__(self, config):
        self.config = config
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'total_frames': 0,
            'saved_frames': 0
        }
    
    def preprocess_roi(self, frame):
        """ROI 추출"""
        if frame is None or frame.size == 0:
            return None
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                
                margin = 0.075
                mx, my = int(w * margin), int(h * margin)
                x, y = max(0, x + mx), max(0, y + my)
                w = min(frame.shape[1] - x, w - 2 * mx)
                h = min(frame.shape[0] - y, h - 2 * my)
                
                if w > 10 and h > 10:
                    return frame[y:y+h, x:x+w]
            
            return frame
        except:
            return frame
    
    def extract_frames(self, video_path):
        """프레임 추출"""
        frames = []
        try:
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total > 0:
                indices = np.linspace(0, total-1, self.config.FRAMES_PER_VIDEO, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        roi = self.preprocess_roi(frame)
                        if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                            resized = cv2.resize(roi, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
                            frames.append(resized)
            cap.release()
        except Exception as e:
            pass
        return frames
    
    def load_json(self, json_path):
        """JSON 로드"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('metas', {})
        except:
            return {}
    
    def prepare_dataset(self):
        """전체 데이터셋 준비"""
        print("="*60)
        print("Step 1: Full Dataset Preparation (Fixing Labels)")
        print("="*60)
        
        # 디렉토리 생성
        self.config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        images_dir = self.config.DATASET_DIR / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # 영상 목록 수집
        video_list = []
        if self.config.VELUM_DIR.exists():
            video_list.extend(list(self.config.VELUM_DIR.glob("*.mp4")))
        if self.config.OTE_DIR.exists():
            video_list.extend(list(self.config.OTE_DIR.glob("*.mp4")))
        
        video_list = sorted(list(set(video_list)))
        self.stats['total_videos'] = len(video_list)
        
        print(f"Processing {len(video_list)} videos...")
        dataset = []
        
        for video_path in tqdm(video_list):
            try:
                json_path = video_path.with_suffix('.json')
                meta = self.load_json(json_path)
                if not meta:
                    self.stats['failed_videos'] += 1
                    continue
                
                phase_str = meta.get('phase', 'Velum')
                cause_str = meta.get('cause', 'no') # 여기서 'Tongue_Base'가 들어옴
                
                # 디버깅용: 이상한 라벨 있으면 출력 (선택사항)
                # if cause_str not in self.config.CAUSE_MAP and cause_str != 'no':
                #    print(f"Unknown label found: {cause_str}")
                
                # 매핑 적용
                if phase_str not in self.config.PHASE_MAP: phase_str = 'Velum'
                
                # 핵심 수정: 매핑 테이블에 없으면 'no' 처리
                target_class = self.config.CAUSE_MAP.get(cause_str, 0)
                
                # 라벨 문자열 통일 (학습 확인용)
                label_name = [k for k, v in self.config.CAUSE_MAP.items() if v == target_class][0]

                frames = self.extract_frames(video_path)
                if not frames:
                    self.stats['failed_videos'] += 1
                    continue
                
                self.stats['total_frames'] += len(frames)
                saved = 0
                
                for i, frame in enumerate(frames):
                    img_name = f"{video_path.stem}_f{i:02d}.jpg"
                    img_path = images_dir / img_name
                    
                    if imwrite_safe(img_path, frame):
                        saved += 1
                        self.stats['saved_frames'] += 1
                        dataset.append({
                            'image_path': f"images/{img_name}",
                            'phase_input': self.config.PHASE_MAP[phase_str],
                            'phase_label': phase_str,
                            'cause_target': target_class,
                            'cause_label': label_name, # 통일된 이름 저장
                            'video_id': video_path.stem
                        })
                
                if saved > 0: self.stats['processed_videos'] += 1
                else: self.stats['failed_videos'] += 1
                    
            except Exception as e:
                self.stats['failed_videos'] += 1
                continue
        
        if not dataset:
            print("\n❌ No data extracted!")
            return None
        
        df = pd.DataFrame(dataset)
        
        print(f"\n{'='*60}")
        print(f"Dataset Re-generated Summary")
        print(f"{'='*60}")
        print(f"Total samples: {len(df)}")
        print(f"\nCause distribution (Check Tongue!):")
        print(df['cause_label'].value_counts())
        
        # Train/Val Split
        try:
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cause_target'])
        except:
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_df.to_csv(self.config.DATASET_DIR / 'train.csv', index=False)
        val_df.to_csv(self.config.DATASET_DIR / 'val.csv', index=False)
        val_df.to_csv(self.config.DATASET_DIR / 'test.csv', index=False)
        
        print(f"\n✅ Data Preparation Complete!")
        return df

def main():
    config = Config()
    if not config.VELUM_DIR.exists() and not config.OTE_DIR.exists():
        print("\n❌ No data directories found!")
        return
    
    DataPreparation(config).prepare_dataset()

if __name__ == "__main__":
    main()
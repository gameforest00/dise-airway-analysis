import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def check_data():
    base_path = Path("../dataset_full")
    train_csv = base_path / "train.csv"
    
    print(f"{'='*60}")
    print(f" 데이터셋 정밀 진단 리포트")
    print(f"{'='*60}")
    
    if not train_csv.exists():
        print(f" 오류: {train_csv} 파일을 찾을 수 없습니다.")
        print("step1_prepare_full.py를 먼저 실행했는지 확인해주세요.")
        return

    # CSV 로드
    df = pd.read_csv(train_csv)
    total_count = len(df)
    
    print(f" 총 학습 데이터 개수: {total_count}개")
    print(f"-"*60)
    
    # 1. 원인(Cause)별 데이터 분포 확인
    print(f"[1] 원인별 데이터 분포 (Cause Distribution)")
    cause_counts = df['cause_label'].value_counts()
    
    for label, count in cause_counts.items():
        ratio = (count / total_count) * 100
        print(f"  • {label:<15}: {count:4d}개 ({ratio:5.1f}%)")
        
    # 시각화 (막대 그래프) -> 팝업으로 뜸
    try:
        plt.figure(figsize=(10, 5))
        cause_counts.plot(kind='bar', color=['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'])
        plt.title(f"Class Distribution (Total: {total_count})")
        plt.xlabel("Cause")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except:
        pass

    print(f"\n" + "-"*60)

    # 2. 페이즈(Phase)별 데이터 분포 확인
    print(f"[2] 촬영 부위별 분포 (Phase Distribution)")
    phase_counts = df['phase_label'].value_counts()
    for label, count in phase_counts.items():
        print(f"  • {label:<15}: {count:4d}개")

    print(f"-"*60)
    
    # 3. 진단 결과
    print(f"[3] 진단 결과")
    
    # OTE 관련 라벨(Tongue, Epiglottis 등)이 있는지 확인
    ote_labels = ['Tongue', 'Epiglottis', 'Oropharynx', 'Tongue_base']
    ote_count = sum(df['cause_label'].isin(ote_labels))
    
    if ote_count == 0:
        print(f" [심각] OTE 관련 데이터가 0개입니다!")
        print(f"   -> 원인: step1에서 JSON 파일을 읽을 때 'cause' 라벨을 못 가져왔거나,")
        print(f"            JSON 파일 안에 해당 라벨이 없을 가능성이 큽니다.")
        print(f"   -> 해결: 원본 JSON 파일을 열어서 'cause' 항목이 어떻게 적혀있는지 확인하세요.")
    
    elif ote_count < total_count * 0.1:
        print(f" [불균형] OTE 데이터가 전체의 {ote_count/total_count*100:.1f}%로 매우 적습니다.")
        print(f"   -> 원인: 데이터 불균형")
        print(f"   -> 해결: 아까 드린 'Class Weight'가 적용된 step2 코드로 학습하면 해결됩니다.")
        
    else:
        print(f" 데이터 비율은 양호합니다. 학습 파라미터나 모델 구조 문제일 수 있습니다.")

if __name__ == "__main__":
    check_data()
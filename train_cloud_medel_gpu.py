import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import glob
import pickle
from sklearn.metrics import average_precision_score, precision_recall_curve


# 앞서 생성한 모듈들 import
sys.path.append('.')
from gk2a_preprocessor import GK2ACloudProcessor
from cloud_convlstm_model import ImprovedCloudMovementPredictor, CloudDataset, improved_train_model

def binary_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-7) -> float:
    """
    pred_mask, true_mask: (B, 1, H, W) 또는 (B, H, W), 값은 0/1
    """
    if pred_mask.dim() == 4 and pred_mask.size(1) == 1:
        pred_mask = pred_mask.squeeze(1)
    if true_mask.dim() == 4 and true_mask.size(1) == 1:
        true_mask = true_mask.squeeze(1)

    intersection = (pred_mask * true_mask).float().sum(dim=(1, 2))
    union = (pred_mask + true_mask - pred_mask * true_mask).float().sum(dim=(1, 2))
    iou = (intersection + eps) / (union + eps)  # (B,)
    return iou.mean().item()

def compute_pr_auc_from_probs(all_probs: np.ndarray, all_targets: np.ndarray) -> float:
    """
    all_probs: (N,) 확률(시그모이드 후)
    all_targets: (N,) 0/1 라벨
    반환: 평균정밀도(AP) = PR-AUC
    """
    # scikit-learn의 average_precision_score는 PR-AUC(AP)를 반환
    return float(average_precision_score(all_targets, all_probs))

class CloudPredictionTrainer:
    """구름 예측 모델 훈련 및 평가 클래스"""

    def __init__(self, data_folder="./processed_cloud_data", model_save_dir="./models"):
        self.data_folder = data_folder
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)

    def prepare_training_data(self, train_ratio=0.8):
        """전처리된 데이터에서 훈련 데이터 준비"""
        print("📊 전처리된 데이터에서 훈련 데이터 준비 중...")

        # 전처리된 파일 리스트 찾기
        processed_files = glob.glob(f"{self.data_folder}/*_processed.pkl")
        processed_files = sorted(processed_files)

        if len(processed_files) < 10:
            raise ValueError(f"충분한 전처리된 데이터가 없습니다. 최소 10개 파일 필요, 현재: {len(processed_files)}개")

        print(f"✅ 총 {len(processed_files)}개 전처리된 파일 발견")

        # 전처리된 데이터 로드
        processed_data = []
        valid_files = []

        for i, filepath in enumerate(processed_files):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                processed_data.append({
                    'filepath': filepath,
                    'binary_mask': data['binary_mask'],
                    'patches': data['patches'],
                    'positions': data['positions'],
                    'timestamp': data.get('metadata', {}).get('time', ''),
                    'filename': data['filename']
                })
                valid_files.append(filepath)

                if (i + 1) % 10 == 0:
                    print(f"로딩 진행률: {i+1}/{len(processed_files)} ({((i+1)/len(processed_files)*100):.1f}%)")

            except Exception as e:
                print(f"⚠️  파일 로딩 실패 {filepath}: {e}")
                continue

        print(f"✅ 데이터 로딩 완료: {len(processed_data)}개 시간대 데이터")

        # 훈련/검증 데이터 분할
        split_idx = int(len(processed_data) * train_ratio)
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]

        return train_data, val_data, valid_files

    def create_data_loaders(self, train_data, val_data, batch_size=32, sequence_length=4):
        """데이터 로더 생성"""
        print("🔄 데이터 로더 생성 중...")

        train_dataset = CloudDataset(
            [d['filepath'] for d in train_data], 
            sequence_length=sequence_length
        )
        val_dataset = CloudDataset(
            [d['filepath'] for d in val_data], 
            sequence_length=sequence_length
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        print(f"✅ 훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")

        return train_loader, val_loader


    def train_and_evaluate(self, train_loader, val_loader, num_epochs=50, device=None,
                       resume=True, best_ckpt_path=None):
        """
        - 실행 시점에 '베스트 가중치(state_dict)만' 로드(있으면).
        - 옵티마이저/스케줄러/에폭은 로드하지 않음 → 항상 Epoch 1부터 시작.
        """
        print("🚀 모델 훈련 시작...")

        model = ImprovedCloudMovementPredictor(
            input_size=(256, 256),
            sequence_length=4,
            prediction_steps=1
        )

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🖥️  사용 디바이스: {device}")
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        model = model.to(device)

        # ✅ 베스트 가중치 로드 (있으면)
        if best_ckpt_path is None:
            best_ckpt_path = os.path.join(self.model_save_dir, 'best_cloud_model.pth')
        if resume and os.path.exists(best_ckpt_path):
            try:
                state = torch.load(best_ckpt_path, map_location=device)
                # state가 state_dict라고 가정 (우리가 그 형태로만 저장)
                model.load_state_dict(state, strict=False)
                print(f"✅ 베스트 가중치 로드 성공: {best_ckpt_path}")
            except Exception as e:
                print(f"⚠️ 베스트 가중치 로드 실패(새로 학습): {e}")

        # --- 학습(항상 Epoch 1부터) ---
        train_losses, val_losses = improved_train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, device=device
        )

        # --- 훈련 곡선 저장/표시 ---
        self.plot_training_history(train_losses, val_losses)

        # --- 마지막 모델도 저장(참고용) ---
        final_model_path = os.path.join(self.model_save_dir, 'final_cloud_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ 훈련 완료! 최종 모델 저장: {final_model_path}")

        return model, train_losses, val_losses


    def evaluate_model(self, model, test_loader, device):
        """모델 성능 평가"""
        print("📊 모델 성능 평가 중...")

        model.eval()

        batch_ious = []
        all_probs_list = []
        all_targets_list = []

        # 분류 지표(accuracy/F1 등)를 위한 이진 예측/정답
        all_bin_preds = []
        all_bin_tgts = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)   # target: (B,1,H,W) 가정
                output = model(data)                                # (B,1,H,W) 로짓
                proba = torch.sigmoid(output)                       # (B,1,H,W) 확률

                # --- mIoU 계산(임계값 0.5로 이진화) ---
                pred_mask = (proba > 0.5).float()                   # (B,1,H,W)
                iou = binary_iou(pred_mask, target)                 # 내부에서 squeeze 처리
                batch_ious.append(iou)

                # --- PR-AUC 계산을 위한 확률/정답(벡터) 축적 ---
                all_probs_list.append(proba.detach().cpu().numpy().ravel())
                all_targets_list.append(target.detach().cpu().numpy().ravel())

                # --- 분류 지표(accuracy/F1 등)용 이진 벡터도 함께 축적 ---
                all_bin_preds.append((proba > 0.5).float().cpu().numpy().ravel())
                all_bin_tgts.append(target.float().cpu().numpy().ravel())

        # --- 집계 ---
        mean_iou = float(np.mean(batch_ious)) if batch_ious else 0.0

        all_probs   = np.concatenate(all_probs_list, axis=0)
        all_targets = np.concatenate(all_targets_list, axis=0).astype(np.int32)

        pr_auc = compute_pr_auc_from_probs(all_probs, all_targets)

        # 이진 지표 계산용 (0/1)
        all_preds_bin = np.concatenate(all_bin_preds, axis=0).astype(np.int32)
        all_tgts_bin  = np.concatenate(all_bin_tgts, axis=0).astype(np.int32)

        accuracy = accuracy_score(all_tgts_bin, all_preds_bin)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_tgts_bin, all_preds_bin, average='binary', zero_division=0
        )
        cm = confusion_matrix(all_tgts_bin, all_preds_bin)

        # --- 로그 ---
        print(f"[Eval] mIoU@0.5 = {mean_iou:.4f} | PR-AUC = {pr_auc:.4f}")
        print(f"   정확도 (Accuracy): {accuracy:.4f}")
        print(f"   정밀도 (Precision): {precision:.4f}")
        print(f"   재현율 (Recall): {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")

        results = {
            'mIoU@0.5': mean_iou,
            'PR-AUC': pr_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        return results


    def plot_training_history(self, train_losses, val_losses):
        """훈련 과정 시각화"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History - Cloud Movement Prediction')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.model_save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ 훈련 과정 그래프 저장: {save_path}")

def print_gpu_info():
    """GPU 정보 출력"""
    print("🔍 GPU 정보 확인:")
    print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 버전: {torch.version.cuda}")
        print(f"   GPU 개수: {torch.cuda.device_count()}")
        print(f"   GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("   GPU를 사용할 수 없습니다.")

def run_full_training_pipeline():
    print("🚀 구름 이동 예측 모델 훈련 파이프라인 시작 (GPU 최적화 버전)")
    print("=" * 60)

    # GPU 정보 출력
    print_gpu_info()
    print()

    # GPU/CPU에 따른 설정
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        batch_size = 16  # GPU용 큰 배치 크기
        num_workers = 4
        print("🚀 GPU 모드로 실행합니다.")
    else:
        device = torch.device('cpu')
        batch_size = 4  # CPU용 작은 배치 크기
        num_workers = 2
        print("⚠️  CPU 모드로 실행합니다.")

    trainer = CloudPredictionTrainer()

    try:
        print("📁 전처리된 데이터 확인 중...")
        
        if not os.path.exists('./processed_cloud_data'):
            print("❌ processed_cloud_data 폴더가 없습니다!")
            print("먼저 전처리 스크립트를 실행하세요.")
            return

        processed_files = glob.glob('./processed_cloud_data/*_processed.pkl')
        if len(processed_files) < 10:
            print(f"⚠️  전처리된 데이터가 부족합니다. 현재: {len(processed_files)}개")
            print("더 많은 데이터를 전처리하거나, 더미 데이터로 테스트해보세요.")
            return

        # 실제 전처리된 데이터로 학습
        train_data, val_data, valid_files = trainer.prepare_training_data(train_ratio=0.8)
        train_loader, val_loader = trainer.create_data_loaders(
            train_data, val_data, 
            batch_size=batch_size, 
            sequence_length=4
        )

        model, train_losses, val_losses = trainer.train_and_evaluate(
            train_loader, val_loader, 
            num_epochs=50,
            device=device
        )

        print("✅ 훈련 파이프라인 완료!")

    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    run_full_training_pipeline()

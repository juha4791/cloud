import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# 앞서 생성한 모듈들 import
sys.path.append('.')
from gk2a_preprocessor import GK2ACloudProcessor
from cloud_convlstm_model import CloudMovementPredictor, CloudDataset, train_model


class CloudPredictionTrainer:
    """구름 예측 모델 훈련 및 평가 클래스"""

    def __init__(self, data_folder="./processed_cloud_data", model_save_dir="./models"):
        self.data_folder = data_folder
        self.model_save_dir = model_save_dir
        self.processor = GK2ACloudProcessor(data_folder)

        os.makedirs(model_save_dir, exist_ok=True)

    def prepare_training_data(self, start_date, end_date, train_ratio=0.8):
        """훈련 데이터 준비"""
        print("📊 훈련 데이터 준비 중...")

        # 시계열 파일 리스트 생성
        file_list = self.processor.collect_time_series(start_date, end_date, interval_minutes=10)

        if len(file_list) < 10:
            raise ValueError(f"충분한 데이터가 없습니다. 최소 10개 파일 필요, 현재: {len(file_list)}개")

        print(f"✅ 총 {len(file_list)}개 파일 발견")

        # 데이터 검증 및 전처리
        processed_data = []
        valid_files = []

        for i, filepath in enumerate(file_list):
            try:
                cloud_mask, metadata = self.processor.load_cloud_data(filepath)
                if cloud_mask is not None:
                    binary_mask = self.processor.preprocess_cloud_mask(cloud_mask)

                    # 패치로 분할 (256x256)
                    patches, positions = self.processor.create_image_patches(
                        binary_mask, patch_size=(256, 256), overlap=0.3
                    )

                    processed_data.append({
                        'filepath': filepath,
                        'patches': patches,
                        'positions': positions,
                        'timestamp': metadata.get('time', '')
                    })
                    valid_files.append(filepath)

                if (i + 1) % 10 == 0:
                    print(f"처리 진행률: {i+1}/{len(file_list)} ({((i+1)/len(file_list)*100):.1f}%)")

            except Exception as e:
                print(f"⚠️  파일 처리 실패 {filepath}: {e}")
                continue

        print(f"✅ 전처리 완료: {len(processed_data)}개 시간대 데이터")

        # 훈련/검증 데이터 분할
        split_idx = int(len(processed_data) * train_ratio)
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]

        return train_data, val_data, valid_files

    def create_data_loaders(self, train_data, val_data, batch_size=8, sequence_length=4):
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
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(f"✅ 훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")

        return train_loader, val_loader

    def train_and_evaluate(self, train_loader, val_loader, num_epochs=50):
        """모델 훈련 및 평가"""
        print("🚀 모델 훈련 시작...")

        model = CloudMovementPredictor(
            input_size=(256, 256),
            sequence_length=4,
            prediction_steps=1
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  사용 디바이스: {device}")

        model = model.to(device)

        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            num_epochs=num_epochs, device=device
        )

        self.plot_training_history(train_losses, val_losses)

        final_model_path = os.path.join(self.model_save_dir, 'final_cloud_model.pth')
        torch.save(model.state_dict(), final_model_path)

        config = {
            'model_architecture': 'ConvLSTM',
            'input_size': [256, 256],
            'sequence_length': 4,
            'prediction_steps': 1,
            'num_epochs': num_epochs,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'device': str(device),
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(self.model_save_dir, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ 훈련 완료! 최종 모델 저장: {final_model_path}")

        return model, train_losses, val_losses

    def evaluate_model(self, model, test_loader, device):
        """모델 성능 평가"""
        print("📊 모델 성능 평가 중...")

        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                predictions = (output > 0.5).float()

                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary'
        )

        cm = confusion_matrix(all_targets, all_predictions)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }

        print(f"📊 성능 결과:")
        print(f"   정확도 (Accuracy): {accuracy:.4f}")
        print(f"   정밀도 (Precision): {precision:.4f}")
        print(f"   재현율 (Recall): {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")

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

def run_full_training_pipeline():
    print("🚀 구름 이동 예측 모델 훈련 파이프라인 시작")
    print("=" * 60)

    trainer = CloudPredictionTrainer()

    try:
        end_date = datetime.now().strftime('%Y%m%d%H%M')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d%H%M')

        print(f"📅 데이터 수집 기간: {start_date} ~ {end_date}")

        if not os.path.exists('./processed_cloud_data') or len(os.listdir('./processed_cloud_data')) < 10:
            print("⚠️  충분한 GK2A 데이터가 없습니다.")
            print("먼저 gk2a_downloader.py를 실행하여 데이터를 수집하세요.")
            return

        # 실제 데이터로 대체
        train_data, val_data, valid_files = trainer.prepare_training_data(start_date, end_date)
        train_loader, val_loader = trainer.create_data_loaders(train_data, val_data, batch_size=2, sequence_length=4)

        model, train_losses, val_losses = trainer.train_and_evaluate(
            train_loader, val_loader, num_epochs=50  # 본격 학습 기간 조절 가능
        )

        print("✅ 훈련 파이프라인 완료!")

    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        return



if __name__ == "__main__":
    run_full_training_pipeline()

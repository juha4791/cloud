import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from datetime import datetime

# 1. 수정된 CloudDataset - 실제 데이터 로딩
class CloudDataset(Dataset):
    """완전히 수정된 CloudDataset - 더미 데이터 완전 제거"""
    
    def __init__(self, data_paths, sequence_length=4, prediction_steps=1):
        self.data_paths = sorted(data_paths)
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        
        # 🎯 핵심: 유효한 시퀀스만 미리 생성
        self.valid_sequences = self._create_valid_sequences()
        print(f"✅ 유효한 시퀀스: {len(self.valid_sequences)}개 생성됨")
        
    def _create_valid_sequences(self):
        """유효한 시퀀스만 생성 (더미 데이터 완전 배제)"""
        valid_sequences = []
        
        for i in range(len(self.data_paths) - self.sequence_length - self.prediction_steps + 1):
            input_paths = self.data_paths[i:i + self.sequence_length]
            target_paths = self.data_paths[i + self.sequence_length:i + self.sequence_length + self.prediction_steps]
            
            # 🔍 데이터 유효성 사전 검증
            all_valid = True
            
            # 입력 데이터 검증
            for path in input_paths:
                if not self._is_valid_data_file(path):
                    all_valid = False
                    break
            
            # 타겟 데이터 검증
            if all_valid:
                for path in target_paths:
                    if not self._is_valid_data_file(path):
                        all_valid = False
                        break
            
            # ✅ 모든 파일이 유효한 경우에만 시퀀스 추가
            if all_valid:
                valid_sequences.append((input_paths, target_paths))
            else:
                print(f"⚠️ 유효하지 않은 시퀀스 제외: {input_paths[0]} ~ {target_paths[-1]}")
        
        return valid_sequences
    
    def _is_valid_data_file(self, path):
        """데이터 파일 유효성 검사"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # binary_mask 존재 및 유효성 검사
            if 'binary_mask' not in data:
                return False
                
            binary_mask = data['binary_mask']
            if binary_mask is None:
                return False
                
            # 크기 검사 (너무 작거나 빈 데이터 제외)
            if binary_mask.size == 0:
                return False
                
            # 모든 값이 동일한 경우 제외 (의미없는 데이터)
            if np.all(binary_mask == binary_mask.flat[0]):
                return False
                
            return True
            
        except Exception:
            return False
    
    def _load_and_process_data(self, path):
        """데이터 로드 및 전처리 (더미 데이터 없음)"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        binary_mask = data['binary_mask']
        
        # 결측값 처리
        binary_mask = np.where(binary_mask == -1, 0, binary_mask)
        
        # 정규화 (0-1 범위)
        if binary_mask.max() > 1:
            binary_mask = binary_mask / binary_mask.max()
        
        # 크기 조정 (256x256)
        binary_mask = self._resize_to_target(binary_mask, (256, 256))
        
        return binary_mask.astype(np.float32)
    
    def _resize_to_target(self, data, target_size=(256, 256)):
        """데이터를 목표 크기로 조정"""
        if data.shape == target_size:
            return data
            
        h, w = data.shape
        target_h, target_w = target_size
        
        if h >= target_h and w >= target_w:
            # 중앙 크롭
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            return data[start_h:start_h+target_h, start_w:start_w+target_w]
        else:
            # 패딩
            pad_h = max(0, (target_h - h) // 2)
            pad_w = max(0, (target_w - w) // 2)
            pad_h_end = target_h - h - pad_h
            pad_w_end = target_w - w - pad_w
            return np.pad(data, ((pad_h, pad_h_end), (pad_w, pad_w_end)), mode='constant')
    
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        """더미 데이터 없는 완전한 데이터 로딩"""
        input_paths, target_paths = self.valid_sequences[idx]
        
        # 🎯 입력 데이터 로딩 - 더미 데이터 없음
        input_data = []
        for path in input_paths:
            try:
                processed_data = self._load_and_process_data(path)
                input_data.append(processed_data)
            except Exception as e:
                # 🚨 더미 데이터 생성 없음! 대신 에러 발생
                raise RuntimeError(f"데이터 로딩 실패: {path}, 에러: {e}")
        
        # 🎯 타겟 데이터 로딩 - 더미 데이터 없음  
        target_data = []
        for path in target_paths:
            try:
                processed_data = self._load_and_process_data(path)
                target_data.append(processed_data)
            except Exception as e:
                # 🚨 더미 데이터 생성 없음! 대신 에러 발생
                raise RuntimeError(f"타겟 로딩 실패: {path}, 에러: {e}")
        
        # 텐서 변환
        # numpy array로 먼저 변환 후 tensor로 변환
        input_array = np.array(input_data)
        target_array = np.array(target_data) 
        input_tensor = torch.FloatTensor(input_array).unsqueeze(1)
        target_tensor = torch.FloatTensor(target_array).unsqueeze(1)

        
        return input_tensor, target_tensor

class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell 구현"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                            out_channels=4 * self.hidden_dim,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            bias=self.bias)
    
    def _init_hidden(self, batch_size, image_size, device=None):
        height, width = image_size
        if device is None:
            device = next(self.parameters()).device
        
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """ConvLSTM 구현"""
    
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, 
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dims[i],
                                        kernel_size=self.kernel_sizes[i],
                                        bias=self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i]._init_hidden(batch_size, image_size))
        return init_states
    
    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if not self.return_all_layers:
            layer_output_list = [layer_output_list[-1]]
            last_state_list = [last_state_list[-1]]
        
        return layer_output_list, last_state_list


# 2. 개선된 모델 아키텍처
class ImprovedCloudMovementPredictor(nn.Module):
    """개선된 구름 이동 예측 모델"""
    
    def __init__(self, input_size=(256, 256), sequence_length=4, prediction_steps=1):
        super(ImprovedCloudMovementPredictor, self).__init__()

        
        # ConvLSTM 레이어 - 더 안정적인 설정
        self.convlstm = ConvLSTM(
            input_dim=1,
            hidden_dims=[32, 32],  # 안정적인 2층 구조 유지
            kernel_sizes=[(3, 3), (3, 3)],
            num_layers=2,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        # 드롭아웃 추가
        self.dropout = nn.Dropout2d(0.1)
        
        # 출력 레이어 개선
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(16, 8, kernel_size=3, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(8, prediction_steps, kernel_size=1)
            # Sigmoid 제거 - BCEWithLogitsLoss 사용
        )
        
    def forward(self, x):
        lstm_out, _ = self.convlstm(x)
        last_output = lstm_out[0][:, -1, :, :, :]
        
        # 드롭아웃 적용
        last_output = self.dropout(last_output)
        
        prediction = self.output_layer(last_output)
        return prediction

# 3. 개선된 훈련 함수
def improved_train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """개선된 모델 훈련 함수 — '가중치만' 저장하여 다음 실행에서 에폭은 항상 1부터 시작"""

    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 1) 불균형 대응 (필요시 값 조정: 2.0 -> 5~15)
    pos_weight = torch.tensor([2.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 2) Optimizer/스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, patience_counter = 15, 0

    # ✅ 저장 경로(로더와 통일): 가중치만 저장
    weights_path = os.path.join("models", "best_cloud_model.pth")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # (선택) 참고용 풀 체크포인트 저장 경로
    full_ckpt_path = os.path.join("models", "best_cloud_model_full.pth")

    print(f"🚀 훈련 시작 - 디바이스: {device}")

    for epoch in range(num_epochs):
        # -------------------- Train --------------------
        model.train()
        train_loss_sum, train_batches = 0.0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                output = model(data)            # (B,1,H,W)
                target = target.squeeze(1)      # (B,H,W)
                loss = criterion(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

                if batch_idx % 3 == 0:
                    print(f'Epoch {epoch+1:2d}/{num_epochs}, Batch {batch_idx:3d}/{len(train_loader):3d}, Loss: {loss.item():.6f}')
            except RuntimeError as e:
                print(f"⚠️ 훈련 중 오류: {e}")
                continue

        avg_train = train_loss_sum / max(1, train_batches)
        train_losses.append(avg_train)

        # -------------------- Validation --------------------
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for data, target in val_loader:
                try:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    target = target.squeeze(1)
                    loss = criterion(output, target)
                    val_loss_sum += loss.item()
                    val_batches += 1
                except RuntimeError as e:
                    print(f"⚠️ 검증 중 오류: {e}")
                    continue

        avg_val = val_loss_sum / max(1, val_batches)
        val_losses.append(avg_val)

        # 스케줄러는 '검증 손실'로 에폭마다 한 번
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val)
        cur_lr = optimizer.param_groups[0]['lr']
        if cur_lr < prev_lr:
            print(f"🔻 LR reduced: {prev_lr:.3e} -> {cur_lr:.3e} (val={avg_val:.6f})")

        print(f'Epoch {epoch+1:2d}/{num_epochs} | 📉 Train: {avg_train:.6f} | 📊 Val: {avg_val:.6f} | 📈 LR: {cur_lr:.3e}')

        # -------------------- Best Save --------------------
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0

            # ✅ 1) 가중치(state_dict)만 저장 — 다음 실행에서 에폭은 1부터 시작
            torch.save(model.state_dict(), weights_path)
            print(f'  💾 [BEST-weights] 저장: {weights_path} (val={best_val_loss:.6f})')

            # 2) (선택) 참고용 풀 체크포인트도 저장하고 싶다면 유지
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
            }, full_ckpt_path)
            print(f'  🧩 [BEST-full] 저장: {full_ckpt_path}')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'🛑 Early stopping at epoch {epoch+1}')
            break

        print('-' * 50)

    return train_losses, val_losses


# 4. 수정된 CloudPredictionTrainer 클래스
class ImprovedCloudPredictionTrainer:
    """개선된 구름 예측 모델 훈련 클래스"""
    
    def __init__(self, data_folder="./processed_cloud_data", model_save_dir="./models"):
        self.data_folder = data_folder
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
    def create_improved_data_loaders(self, train_data, val_data, batch_size=4, sequence_length=4):
        """개선된 데이터 로더 생성"""
        print("🔄 개선된 데이터 로더 생성 중...")
        
        # 증강 적용은 훈련 데이터에만
        train_dataset = CloudDataset(
            [d['filepath'] for d in train_data],
            sequence_length=sequence_length,
        )
        
        val_dataset = CloudDataset(
            [d['filepath'] for d in val_data],
            sequence_length=sequence_length,
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✅ 훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(val_dataset)}개")
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, num_epochs=100, device=None):
        """개선된 모델 훈련"""
        print("🚀 개선된 모델 훈련 시작...")
        
        model = ImprovedCloudMovementPredictor(
            input_size=(256, 256),
            sequence_length=4,
            prediction_steps=1
        )
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print(f"🖥️ 사용 디바이스: {device}")
        model = model.to(device)
        
        # 모델 파라미터 수 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 총 파라미터: {total_params:,}")
        print(f"📊 훈련 가능 파라미터: {trainable_params:,}")
        
        train_losses, val_losses = improved_train_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs, device=device
        )
        
        return model, train_losses, val_losses

# 5. 사용법
def run_improved_training_pipeline():
    """개선된 훈련 파이프라인 실행"""
    print("🚀 개선된 구름 이동 예측 모델 훈련 파이프라인 시작")
    print("=" * 60)
    
    # GPU/CPU 설정
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        batch_size = 4  # 메모리 안정성을 위해 작게 설정
        print("🚀 GPU 모드로 실행")
    else:
        device = torch.device('cpu')
        batch_size = 2
        print("⚠️ CPU 모드로 실행")
    
    trainer = ImprovedCloudPredictionTrainer()
    
    try:
        # 전처리된 데이터 확인
        processed_files = []
        if os.path.exists('./processed_cloud_data'):
            import glob
            processed_files = glob.glob('./processed_cloud_data/*_processed.pkl')
        
        if len(processed_files) < 10:
            print(f"❌ 전처리된 데이터 부족: {len(processed_files)}개")
            print("먼저 gk2a_preprocessor.py를 실행하세요.")
            return
        
        # 훈련/검증 데이터 분할
        split_idx = int(len(processed_files) * 0.8)
        train_files = processed_files[:split_idx]
        val_files = processed_files[split_idx:]
        
        train_data = [{'filepath': f} for f in train_files]
        val_data = [{'filepath': f} for f in val_files]
        
        print(f"📊 훈련 파일: {len(train_files)}개")
        print(f"📊 검증 파일: {len(val_files)}개")
        
        # 데이터 로더 생성
        train_loader, val_loader = trainer.create_improved_data_loaders(
            train_data, val_data,
            batch_size=batch_size,
            sequence_length=4
        )
        
        # 모델 훈련
        model, train_losses, val_losses = trainer.train_improved_model(
            train_loader, val_loader,
            num_epochs=100,
            device=device
        )
        
        print("✅ 개선된 훈련 파이프라인 완료!")
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    run_improved_training_pipeline()

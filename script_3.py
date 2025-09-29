# ConvLSTM 모델 구현을 위한 코드 템플릿 생성
convlstm_code = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from datetime import datetime

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
    """ConvLSTM 네트워크"""
    
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
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dims[i],
                                        kernel_size=self.kernel_sizes[i],
                                        bias=self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)

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
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                cur_state=[h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i]._init_hidden(batch_size, image_size))
        return init_states

class CloudMovementPredictor(nn.Module):
    """구름 이동 예측 모델"""
    
    def __init__(self, input_size=(256, 256), sequence_length=4, prediction_steps=1):
        super(CloudMovementPredictor, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        
        # ConvLSTM 레이어
        self.convlstm = ConvLSTM(
            input_dim=1,
            hidden_dims=[32, 32],
            kernel_sizes=[(3, 3), (3, 3)],
            num_layers=2,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, prediction_steps, kernel_size=1),
            nn.Sigmoid()  # 구름 확률 (0-1)
        )
        
    def forward(self, x):
        # x: (batch, sequence, channels, height, width)
        lstm_out, _ = self.convlstm(x)
        
        # 마지막 시간 스텝의 출력 사용
        last_output = lstm_out[0][:, -1, :, :, :]  # (batch, channels, height, width)
        
        # 예측 출력
        prediction = self.output_layer(last_output)
        
        return prediction

class CloudDataset(Dataset):
    """구름 데이터셋 클래스"""
    
    def __init__(self, data_paths, sequence_length=4, prediction_steps=1):
        self.data_paths = sorted(data_paths)
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        
        # 시계열 시퀀스 생성
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        sequences = []
        for i in range(len(self.data_paths) - self.sequence_length - self.prediction_steps + 1):
            input_sequence = self.data_paths[i:i + self.sequence_length]
            target = self.data_paths[i + self.sequence_length:i + self.sequence_length + self.prediction_steps]
            sequences.append((input_sequence, target))
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_paths, target_paths = self.sequences[idx]
        
        # 입력 시퀀스 로드
        input_data = []
        for path in input_paths:
            # 여기서 실제 데이터 로딩 로직 구현
            # 예: cloud_mask = load_cloud_data(path)
            # 임시로 더미 데이터 생성
            dummy_data = np.random.randint(0, 2, (256, 256)).astype(np.float32)
            input_data.append(dummy_data)
        
        # 타겟 데이터 로드
        target_data = []
        for path in target_paths:
            dummy_target = np.random.randint(0, 2, (256, 256)).astype(np.float32)
            target_data.append(dummy_target)
        
        input_tensor = torch.FloatTensor(input_data).unsqueeze(1)  # (seq, 1, H, W)
        target_tensor = torch.FloatTensor(target_data)  # (pred_steps, H, W)
        
        return input_tensor, target_tensor

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """모델 훈련 함수"""
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # 타겟을 (batch, pred_steps, H, W) 형태로 변환
            target = target.permute(1, 0, 2, 3)  # (pred_steps, batch, H, W) -> (batch, pred_steps, H, W)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                target = target.permute(1, 0, 2, 3)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_cloud_model.pth')
            print(f'✅ 최고 성능 모델 저장 (Val Loss: {best_val_loss:.6f})')
    
    return train_losses, val_losses

# 사용 예시
if __name__ == "__main__":
    # 모델 초기화
    model = CloudMovementPredictor(
        input_size=(256, 256),
        sequence_length=4,
        prediction_steps=1
    )
    
    print(f"✅ 모델 생성 완료")
    print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # GPU 사용 가능 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  사용 디바이스: {device}")
    
    model = model.to(device)
    
    # 더미 데이터로 모델 테스트
    dummy_input = torch.randn(2, 4, 1, 256, 256).to(device)  # (batch, seq, channels, H, W)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ 모델 테스트 완료 - 출력 크기: {output.shape}")
'''

# ConvLSTM 모델 코드 저장
with open('cloud_convlstm_model.py', 'w', encoding='utf-8') as f:
    f.write(convlstm_code)

print("✅ ConvLSTM 모델 코드 생성 완료: cloud_convlstm_model.py")
print("\n📦 필요한 패키지:")
print("pip install torch torchvision scikit-learn")
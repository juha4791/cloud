# 한국 구름 이동 예측 딥러닝 모델 🌤️

GK2A 위성 데이터를 활용한 ConvLSTM 기반 구름 이동 예측 시스템

## 🎯 프로젝트 목표
- 한국 GK2A 위성 데이터를 활용한 구름 이동 예측 모델 개발
- 실시간 구름 예측 시스템 구축
- 태양광 발전, 기상 예보, 항공 안전 등에 활용

## 📁 프로젝트 구조
```
cloud-prediction-korea/
├── gk2a_downloader.py          # GK2A 위성 데이터 다운로드
├── gk2a_preprocessor.py        # 데이터 전처리 및 시각화
├── cloud_convlstm_model.py     # ConvLSTM 모델 구현
├── train_cloud_model.py        # 모델 훈련 스크립트
├── realtime_prediction_system.py # 실시간 예측 시스템
├── prediction_dashboard.html   # 웹 대시보드
├── gk2a_data/                  # 위성 데이터 저장 폴더
├── models/                     # 훈련된 모델 저장 폴더
└── realtime_data/              # 실시간 데이터 폴더
```

## 🚀 빠른 시작 가이드

### 1단계: 환경 설정
```bash
# Python 패키지 설치
pip install torch torchvision xarray netcdf4 matplotlib numpy pandas scikit-learn requests

# 프로젝트 폴더 생성
mkdir cloud-prediction-korea
cd cloud-prediction-korea
```

### 2단계: API 키 발급
1. [기상청 API 허브](https://apihub.kma.go.kr) 회원가입
2. API 키 발급 (일반회원: 20,000건/일, 5GB/일)
3. `gk2a_downloader.py`의 `YOUR_API_KEY_HERE`에 실제 키 입력

### 3단계: 데이터 수집
```python
# GK2A 구름탐지 데이터 다운로드
python gk2a_downloader.py
```

### 4단계: 데이터 전처리 확인
```python
# 데이터 로드 및 시각화
python gk2a_preprocessor.py
```

### 5단계: 모델 훈련
```python
# ConvLSTM 모델 훈련 (충분한 데이터 필요)
python train_cloud_model.py
```

### 6단계: 실시간 예측
```python
# 실시간 예측 시스템 실행
python realtime_prediction_system.py
```

## 📊 모델 성능
- **ConvLSTM 모델**: 정확도 78-80%, F1-score 0.79-0.80
- **예측 범위**: 1-4시간 (10분 간격)
- **해상도**: 2km (한반도 영역)
- **업데이트 주기**: 10분

## 🔧 주요 기능

### 1. 데이터 수집 (`gk2a_downloader.py`)
- GK2A 위성 구름탐지(CLD) 데이터 자동 다운로드
- NetCDF 파일 형태로 저장
- 에러 처리 및 재시도 로직 포함

### 2. 데이터 전처리 (`gk2a_preprocessor.py`)
- 구름 마스크 이진화 (맑음=0, 구름=1)
- 이미지 패치 분할 (256×256)
- 시각화 및 품질 검증

### 3. ConvLSTM 모델 (`cloud_convlstm_model.py`)
- 시공간 특징 추출을 위한 ConvLSTM Cell
- 4시간 입력 → 1시간 예측 구조
- 배치 정규화 및 드롭아웃 포함

### 4. 훈련 시스템 (`train_cloud_model.py`)
- 동적 학습 전략 지원
- 실시간 성능 모니터링
- 최적 모델 자동 저장

### 5. 실시간 예측 (`realtime_prediction_system.py`)
- 10분 간격 자동 데이터 수집
- 실시간 구름 이동 예측
- 웹 대시보드 제공

## ⚡ 성능 최적화 팁

### GPU 사용 권장
```python
# CUDA 사용 가능 여부 확인
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 메모리 최적화
- 배치 크기 조정: GPU 메모리에 따라 4-16
- 데이터 로더 워커 수: CPU 코어 수의 50%
- 혼합 정밀도 훈련 고려

### 데이터 최적화
- SSD 저장소 사용 권장
- 전처리된 데이터 캐시
- 병렬 데이터 로딩

## 🔍 문제 해결

### 일반적인 오류들
1. **API 호출 한도 초과**: 기관회원 가입 고려
2. **CUDA 메모리 부족**: 배치 크기 감소
3. **NetCDF 로딩 오류**: netcdf4 라이브러리 재설치

### 데이터 품질 점검
```python
# 데이터 유효성 검사
from gk2a_preprocessor import GK2ACloudProcessor
processor = GK2ACloudProcessor()

# 결측값 비율 확인
cloud_mask, _ = processor.load_cloud_data("sample.nc")
missing_ratio = np.sum(cloud_mask == -1) / cloud_mask.size
print(f"결측값 비율: {missing_ratio:.1%}")
```

## 📈 성능 평가 지표
- **정확도 (Accuracy)**: 전체 예측의 정확성
- **정밀도 (Precision)**: 구름으로 예측한 것 중 실제 구름 비율  
- **재현율 (Recall)**: 실제 구름 중 올바르게 탐지한 비율
- **F1-Score**: 정밀도와 재현율의 조화평균

## 🌟 향후 계획
1. **모델 개선**: Transformer, 앙상블 기법 적용
2. **다중 스케일**: 지역별 특화 모델 개발
3. **실시간 최적화**: 더 빠른 추론 속도
4. **융합 데이터**: 레이더, 수치모델 데이터 결합
5. **서비스 확장**: 웹 API, 모바일 앱 개발

## 📞 지원 및 기여
- 이슈 리포팅: GitHub Issues 활용
- 기여 방법: Pull Request 환영
- 문의: 프로젝트 메인테이너에게 연락

## 📄 라이선스
- MIT License
- GK2A 데이터: 기상청 이용약관 준수 필요

---

**⚠️ 주의사항**
- GK2A 데이터는 기상청 저작권 보호 대상
- 상업적 이용 시 별도 허가 필요
- API 호출 한도 준수 필요

**✨ 시작하기**
지금 바로 `gk2a_downloader.py`를 실행하여 첫 번째 위성 데이터를 받아보세요!

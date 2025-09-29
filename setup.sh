#!/bin/bash
# 한국 구름 예측 모델 설치 스크립트

echo "🌤️  한국 구름 이동 예측 모델 설치 시작"
echo "======================================"

# Python 가상환경 생성 (선택사항)
echo "📦 Python 가상환경 생성..."
python -m venv cloud_prediction_env
source cloud_prediction_env/bin/activate  # Windows: cloud_prediction_env\Scripts\activate

# 필요 패키지 설치
echo "📥 필요 패키지 설치..."
pip install torch torchvision torchaudio
pip install xarray netcdf4 matplotlib numpy pandas scikit-learn requests

# 데이터 폴더 생성
echo "📁 데이터 폴더 생성..."
mkdir -p gk2a_data models realtime_data

echo "✅ 설치 완료!"
echo ""
echo "다음 단계:"
echo "1. 기상청 API 허브에서 API 키 발급: https://apihub.kma.go.kr"
echo "2. gk2a_downloader.py에 API 키 입력"
echo "3. python gk2a_downloader.py 실행"
echo ""
echo "🚀 즐거운 연구 되세요!"

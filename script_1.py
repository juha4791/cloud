# GK2A 구름탐지 데이터 다운로드를 위한 Python 코드 템플릿 생성
sample_code = '''
import requests
import os
from datetime import datetime, timedelta

def download_gk2a_cloud_data(date_str, api_key, save_folder="./gk2a_data"):
    """
    GK2A 구름탐지(CLD) 데이터 다운로드
    
    Args:
        date_str: 'YYYYMMDDHHMM' 형식 (예: '202409240600')
        api_key: 기상청 API 인증키
        save_folder: 저장할 폴더 경로
    """
    
    # 한반도(KO) 영역 구름탐지 데이터 URL
    url = f"https://apihub.kma.go.kr/api/typ05/api/GK2A/LE2/CLD/KO/data?date={date_str}&authKey={api_key}"
    
    # 저장 폴더 생성
    os.makedirs(save_folder, exist_ok=True)
    
    # 파일명 생성
    filename = f"gk2a_ami_le2_cld_ko020lc_{date_str}.nc"
    filepath = os.path.join(save_folder, filename)
    
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ 다운로드 완료: {filename} ({len(response.content)/1024/1024:.2f} MB)")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 다운로드 실패: {e}")
        return None

# 사용 예시
if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY_HERE"  # 발급받은 API 키 입력
    
    # 최근 데이터 다운로드 (2분 간격)
    now = datetime.now() - timedelta(hours=2)  # 2시간 전 데이터
    date_str = now.strftime("%Y%m%d%H%M")
    
    filepath = download_gk2a_cloud_data(date_str, API_KEY)
    
    if filepath:
        print(f"다운로드된 파일: {filepath}")
'''

# 코드를 파일로 저장
with open('gk2a_downloader.py', 'w', encoding='utf-8') as f:
    f.write(sample_code)

print("✅ GK2A 데이터 다운로드 코드 생성 완료: gk2a_downloader.py")
print("\n📋 다음 단계:")
print("1. 기상청 API 허브에서 API 키 발급")
print("2. 코드의 YOUR_API_KEY_HERE 부분에 실제 API 키 입력")
print("3. python gk2a_downloader.py 실행")
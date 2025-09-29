import os
import requests
from datetime import datetime, timedelta

BASE_URL = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE2/CLD/KO/data"

def download_gk2a_cloud_data(date_str: str, api_key: str, save_folder: str = "./gk2a_data") -> str | None:
    """
    GK2A 구름탐지(CLD) 데이터 다운로드
    - date_str: 'YYYYMMDDHHMM' (예: '202509012300')
    - api_key : 기상청 API 인증키
    """
    os.makedirs(save_folder, exist_ok=True)

    # ✅ 인자 반영: 날짜/키를 쿼리에 실제로 넣기
    params = {
        "date": date_str,
        "authKey": api_key,
    }

    # 저장 파일명
    filename = f"gk2a_ami_le2_cld_ko020lc_{date_str}.nc"
    filepath = os.path.join(save_folder, filename)

    try:
        r = requests.get(BASE_URL, params=params, timeout=300)
        r.raise_for_status()

        content = r.content
        # 간단한 유효성 검사: 크기가 너무 작으면 실패 처리
        if len(content) < 10_000:
            print(f"⚠️ 응답 크기가 비정상적으로 작습니다({len(content)} bytes). date={date_str}")
            return None

        with open(filepath, "wb") as f:
            f.write(content)

        print(f"✅ 다운로드 완료: {filename} ({len(content)/1024/1024:.2f} MB)")
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"❌ 다운로드 실패[{date_str}]: {e}")
        return None


if __name__ == "__main__":
    API_KEY = os.getenv("KMA_API_KEY", "").strip() or "3eMe99VHTnSjHvfVR_502g"
    # 예시: 최근 2일치 10분 간격
    end_time = datetime.utcnow() - timedelta(hours=1)
    start_time = end_time - timedelta(days=2)

    cur = start_time
    while cur <= end_time:
        date_str = cur.strftime("%Y%m%d%H%M")
        path = download_gk2a_cloud_data(date_str, API_KEY)
        if not path:
            print(f"다운로드 실패: {date_str}")
        cur += timedelta(minutes=10)

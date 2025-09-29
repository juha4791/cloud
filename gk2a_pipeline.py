# ======================= gk2a_pipeline.py =======================
import os
import requests
import pickle
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset

# ------------------ 기본 설정 ------------------
BASE_URL = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE2/CLD/KO/data"
SAVE_FOLDER = "./processed_cloud_data"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# ------------------ 유틸 함수 ------------------
def round_to_10min(dt: datetime) -> datetime:
    """UTC datetime을 10분 단위로 내림"""
    minute = (dt.minute // 10) * 10
    return dt.replace(minute=minute, second=0, microsecond=0)

def load_binary_mask_from_pkl(path: str) -> np.ndarray | None:
    """_processed.pkl에서 binary_mask 불러오기"""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data.get("binary_mask")
    except Exception:
        return None

# ------------------ 1. 다운로드 ------------------
def download_gk2a_cloud_data(date_str: str, api_key: str) -> str | None:
    """GK2A 구름탐지(CLD) 데이터 다운로드"""
    params = {"date": date_str, "authKey": api_key}
    filename = f"gk2a_ami_le2_cld_ko020lc_{date_str}.nc"
    filepath = os.path.join(SAVE_FOLDER, filename)

    try:
        r = requests.get(BASE_URL, params=params, timeout=300)
        r.raise_for_status()
        content = r.content

        if len(content) < 10_000:
            print(f"⚠️ 응답 크기 비정상({len(content)} bytes). date={date_str}")
            return None

        with open(filepath, "wb") as f:
            f.write(content)

        print(f"✅ 다운로드 완료: {filename} ({len(content)/1024/1024:.2f} MB)")
        return filepath

    except requests.exceptions.RequestException as e:
        print(f"❌ 다운로드 실패[{date_str}]: {e}")
        return None

# ------------------ 2. 전처리 ------------------
def preprocess_nc_to_pkl(nc_path: str) -> str | None:
    """nc 파일에서 binary_mask + patches + metadata 추출 후 _processed.pkl 저장"""
    try:
        ds = Dataset(nc_path, "r")
        # CLD 제품은 "CLD" 또는 "cld" 변수 포함
        var_name = None
        for cand in ["CLD", "cld", "CloudMask"]:
            if cand in ds.variables:
                var_name = cand
                break
        if var_name is None:
            print(f"⚠️ 변수(CL D mask)를 찾을 수 없음: {nc_path}")
            return None

        data = ds.variables[var_name][:].astype(np.float32)
        ds.close()

        # 단순 이진화 (값 1=구름, 0=없음 / -1=결측)
        binary_mask = np.where(data == 1, 1, 0).astype(np.float32)

        # ✅ 패치 나누기 (256x256 크기)
        H, W = binary_mask.shape
        patch_size = 256
        patches = []
        positions = []
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                patch = binary_mask[i:i+patch_size, j:j+patch_size]
                if patch.shape == (patch_size, patch_size):
                    patches.append(patch)
                    positions.append((i, j))
        patches = np.stack(patches, axis=0)

        # ✅ 메타데이터
        fname = os.path.basename(nc_path)
        date_str = fname.split("_")[5].split(".")[0]  # YYYYMMDDHHMM
        metadata = {"time": date_str}

        out_dict = {
            "binary_mask": binary_mask,
            "patches": patches,
            "positions": positions,
            "filename": fname,
            "metadata": metadata,
        }

        out_path = nc_path.replace(".nc", "_processed.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f)

        print(f"✅ 전처리 완료: {os.path.basename(out_path)} "
              f"(cover={binary_mask.mean()*100:.2f}%, patches={len(patches)})")
        return out_path
    except Exception as e:
        print(f"❌ 전처리 실패: {nc_path} | {e}")
        return None

# ------------------ 3. 중복 검사 ------------------
def check_duplicate_and_cleanup(new_pkl: str) -> bool:
    """binary_mask 배열 단위 중복 검사"""
    mask_new = load_binary_mask_from_pkl(new_pkl)
    if mask_new is None:
        return False

    for fn in os.listdir(SAVE_FOLDER):
        if fn.endswith("_processed.pkl") and fn != os.path.basename(new_pkl):
            mask_exist = load_binary_mask_from_pkl(os.path.join(SAVE_FOLDER, fn))
            if mask_exist is not None and np.array_equal(mask_new, mask_exist):
                # 중복 발견 시 원본 nc + pkl 삭제
                nc_file = new_pkl.replace("_processed.pkl", ".nc")
                if os.path.exists(nc_file):
                    os.remove(nc_file)
                os.remove(new_pkl)
                print(f"⚠️ 중복 배열 감지 → 삭제: {os.path.basename(new_pkl)}")
                return True
    return False

# ------------------ 메인 실행 ------------------
if __name__ == "__main__":
    API_KEY = os.getenv("KMA_API_KEY", "").strip() or "3eMe99VHTnSjHvfVR_502g"

    # 최근 6시간치 예시
    end_time = round_to_10min(datetime.utcnow() - timedelta(hours=1))
    start_time = end_time - timedelta(hours=6)

    cur = start_time
    while cur <= end_time:
        date_str = cur.strftime("%Y%m%d%H%M")

        # 1) 다운로드
        nc_path = download_gk2a_cloud_data(date_str, API_KEY)
        if not nc_path:
            cur += timedelta(minutes=10)
            continue

        # 2) 전처리
        pkl_path = preprocess_nc_to_pkl(nc_path)
        if not pkl_path:
            cur += timedelta(minutes=10)
            continue

        # 3) 중복 검사
        check_duplicate_and_cleanup(pkl_path)

        cur += timedelta(minutes=10)

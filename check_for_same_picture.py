import os, pickle, hashlib, numpy as np
from glob import glob

# ==== 여기에 확인할 pkl 경로들을 넣으세요 ====
files = [
    r"./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202509012300_processed.pkl",
    r"./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202509030640_processed.pkl",
    r"./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202509041510_processed.pkl",
    r"./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202509200440_processed.pkl",
    r"./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202509211920_processed.pkl",
    r"./processed_cloud_data/gk2a_ami_le2_cld_ko020lc_202509231410_processed.pkl",
]

def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def load_mask(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # 확률 채널 우선
    for k in ["prob_mask","cloud_prob","probability","prob"]:
        if k in data:
            arr = np.asarray(data[k], dtype=np.float32)
            break
    else:
        arr = np.asarray(data["binary_mask"], dtype=np.float32)
    arr = np.where(arr==-1, 0, arr)
    if arr.max() > 1: arr = arr / arr.max()
    return data, arr

def coverage(a, thr=0.5):
    b = (a >= thr).astype(np.float32)
    return float(b.mean()*100.0)

# 1) 파일 단위, 배열 단위 해시/통계
infos = []
for p in files:
    if not os.path.exists(p):
        print(f"[MISS] {p} (없음)")
        continue
    file_md5 = md5_bytes(open(p,"rb").read())
    data, arr = load_mask(p)
    arr_md5 = md5_bytes(arr.tobytes())
    cov = coverage(arr)
    uniq = int(np.unique(arr).size)
    print(f"[FILE] {os.path.basename(p)}")
    print(f"  file_md5={file_md5}")
    print(f"  arr_md5 ={arr_md5}")
    print(f"  shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, uniq={uniq}, cover={cov:.2f}%")
    infos.append((p, file_md5, arr_md5, arr))

# 2) 프레임 간 완전 동일/거의 동일 여부
def near_same(a,b,eps=1e-6):
    return float(np.mean(np.abs(a-b))) < eps

for i in range(len(infos)-1):
    p1, _, h1, a1 = infos[i]
    p2, _, h2, a2 = infos[i+1]
    same_hash = (h1==h2)
    mae = float(np.mean(np.abs(a1-a2)))
    xor = int(( (a1>=0.5)^(a2>=0.5) ).sum())
    print(f"[PAIR] {os.path.basename(p1)} vs {os.path.basename(p2)}")
    print(f"  arr_md5_equal={same_hash}, mean_abs_diff={mae:.8f}, xor_pixels={xor}")

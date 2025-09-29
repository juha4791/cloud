import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os, pickle, hashlib, json
from datetime import datetime, timedelta

class GK2ACloudProcessor:
    """GK2A 구름 데이터 전처리 클래스"""

    def __init__(self, data_folder="./gk2a_data"):
        self.data_folder = data_folder

    def _md5(self, a: np.ndarray) -> str:
        return hashlib.md5(a.tobytes()).hexdigest()

    def load_cloud_data(self, filepath):
        """NetCDF 파일 로드 및 구름 데이터 추출"""
        try:
            ds = xr.open_dataset(filepath)

            # GK2A CLD: 0=맑음, 1=구름가능성, 2=구름, 3=확실한 구름
            if 'CLD' not in ds.variables:
                raise RuntimeError("CLD 변수 없음")

            cld = ds['CLD'].values  # (H,W)
            # 결측(NaN) -> -1 치환
            cld = np.where(np.isfinite(cld), cld, -1).astype(np.int16)

            # 메타데이터
            metadata = {
                'observation_start_time': ds.attrs.get('observation_start_time', ''),
                'projection': ds.attrs.get('projection', ''),
                'resolution': ds.attrs.get('resolution', ''),
                'lat_range': [ds.attrs.get('geospatial_lat_min', 0), ds.attrs.get('geospatial_lat_max', 0)],
                'lon_range': [ds.attrs.get('geospatial_lon_min', 0), ds.attrs.get('geospatial_lon_max', 0)]
            }
            ds.close()
            return cld, metadata

        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            return None, None

    def build_prob_and_binary(self, cld: np.ndarray):
        """
        - 확률채널(연속): CLD를 0~1로 선형 스케일 (임시 기준)
          * -1(결측)은 0으로 처리
        - 이진화: CLD >= 2 -> 1, 그 외 0; 결측은 0 처리
        """
        cld_clean = cld.copy()
        cld_clean = np.where(cld_clean == -1, 0, cld_clean)
        # 0..3 범위를 0..1로 스케일 (간단히 /3)
        prob = np.clip(cld_clean.astype(np.float32) / 3.0, 0.0, 1.0)

        binary = (cld_clean >= 2).astype(np.float32)  # 0/1
        return prob, binary

    def create_image_patches(self, arr, patch_size=(256, 256), overlap=0.5):
        """이미지를 패치로 분할"""
        h, w = arr.shape
        step = int(patch_size[0] * (1 - overlap))
        patches, positions = [], []
        for i in range(0, h - patch_size[0] + 1, step):
            for j in range(0, w - patch_size[1] + 1, step):
                patch = arr[i:i+patch_size[0], j:j+patch_size[1]]
                patches.append(patch)
                positions.append((i, j))
        return np.array(patches), positions

    def visualize_cloud_data(self, cloud_mask, title="GK2A Cloud Mask", save_path=None):
        """구름 데이터 시각화(이진/정수 CLD)"""
        plt.figure(figsize=(10, 8))
        vmax = max(3, int(np.nanmax(cloud_mask)))
        colors = ['white', 'lightblue', 'blue', 'darkblue', 'gray']
        cmap = ListedColormap(colors[:vmax+2])
        plt.imshow(cloud_mask, cmap=cmap, vmin=-1, vmax=vmax)
        plt.colorbar(label='CLD (-1,0..3)')
        plt.title(title)
        plt.xlabel('X'); plt.ylabel('Y')
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✅ 이미지 저장: {save_path}")
        plt.close()

    def collect_time_series(self, start_date, end_date, interval_minutes=10):
        file_list = []
        cur = datetime.strptime(start_date, '%Y%m%d%H%M')
        end = datetime.strptime(end_date, '%Y%m%d%H%M')
        while cur <= end:
            date_str = cur.strftime('%Y%m%d%H%M')
            filename = f"gk2a_ami_le2_cld_ko020lc_{date_str}.nc"
            fp = os.path.join(self.data_folder, filename)
            if os.path.exists(fp):
                file_list.append(fp)
            cur += timedelta(minutes=interval_minutes)
        return file_list


if __name__ == "__main__":
    import glob
    data_folder = r"C:\Users\hwhhs\Desktop\cloud\gk2a_data"
    output_folder = r"C:\Users\hwhhs\Desktop\cloud\processed_cloud_data"
    os.makedirs(output_folder, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(data_folder, "*.nc")))
    print(f"📁 총 {len(all_files)}개 파일 발견")
    print(f"💾 전처리 결과 저장 폴더: {output_folder}")

    if not all_files:
        print("❌ .nc 파일이 없습니다!")
        raise SystemExit

    processed_count = failed_count = 0
    dup_guard = {}

    for i, filepath in enumerate(all_files):
        try:
            filename = os.path.basename(filepath).replace(".nc", "")
            print(f"\n🔄 처리 중 ({i+1}/{len(all_files)}): {filename}")

            proc = GK2ACloudProcessor(data_folder)
            cld, metadata = proc.load_cloud_data(filepath)
            if cld is None:
                failed_count += 1
                continue

            print(f"✅ 데이터 로드: shape={cld.shape}, unique={np.unique(cld).tolist()[:10]} ...")
            prob, binary = proc.build_prob_and_binary(cld)

            # 패치(필요 시)
            patches, positions = proc.create_image_patches(binary)

            # 메타/해시/커버율
            prob_md5 = proc._md5(prob)
            bin_md5 = proc._md5(binary)
            coverage = float(binary.mean() * 100.0)

            meta = {
                "original_filepath": filepath,
                "filename": filename,
                "created_at": datetime.now().isoformat(),
                "shape": list(cld.shape),
                "coverage%": coverage,
                "prob_md5": prob_md5,
                "bin_md5": bin_md5,
                "metadata": metadata,
            }

            # 저장(.pkl)
            save_path = os.path.join(output_folder, f"{filename}_processed.pkl")
            with open(save_path, "wb") as f:
                pickle.dump({
                    "filename": filename,
                    "cloud_prob": prob.astype(np.float32),   # ✅ 연속 확률 채널
                    "binary_mask": binary.astype(np.float32), # ✅ 이진 마스크(0/1)
                    "patches": patches,
                    "positions": positions,
                    "meta": meta
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"💾 저장: {save_path} | cover={coverage:.2f}%")

            # 중복 경보: 같은 bin_md5가 너무 자주 나오면 표시
            dup_guard[bin_md5] = dup_guard.get(bin_md5, 0) + 1
            if dup_guard[bin_md5] >= 3:
                print(f"⚠️ 동일 이진 마스크가 {dup_guard[bin_md5]}회 관측됨(bin_md5={bin_md5[:8]}...)")

            # 시각화
            img_path = os.path.join(output_folder, f"{filename}_visual.png")
            proc.visualize_cloud_data(cld, title=f"CLD (raw) - {filename}", save_path=img_path)

            processed_count += 1

        except Exception as e:
            print(f"❌ 처리 실패: {e}")
            failed_count += 1

    print("\n🎉 일괄 처리 완료!")
    print(f"✅ 성공: {processed_count}개 | ❌ 실패: {failed_count}개")
    print(f"📁 저장 위치: {output_folder}")

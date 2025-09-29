# NetCDF 파일 처리를 위한 전처리 코드 템플릿 생성
preprocessing_code = '''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
from datetime import datetime, timedelta

class GK2ACloudProcessor:
    """GK2A 구름 데이터 전처리 클래스"""
    
    def __init__(self, data_folder="./gk2a_data"):
        self.data_folder = data_folder
        
    def load_cloud_data(self, filepath):
        """NetCDF 파일 로드 및 구름 데이터 추출"""
        try:
            ds = xr.open_dataset(filepath)
            
            # 구름탐지 데이터 추출 (CLD 변수)
            cloud_mask = ds['CLD'].values
            
            # 메타데이터 추출
            metadata = {
                'time': ds.attrs.get('observation_start_time', ''),
                'projection': ds.attrs.get('projection', ''),
                'resolution': ds.attrs.get('resolution', ''),
                'lat_range': [ds.attrs.get('geospatial_lat_min', 0), 
                             ds.attrs.get('geospatial_lat_max', 0)],
                'lon_range': [ds.attrs.get('geospatial_lon_min', 0), 
                             ds.attrs.get('geospatial_lon_max', 0)]
            }
            
            ds.close()
            return cloud_mask, metadata
            
        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            return None, None
    
    def preprocess_cloud_mask(self, cloud_mask):
        """구름 마스크 전처리 (이진화)"""
        # GK2A CLD 값: 0=맑음, 1=구름가능성, 2=구름, 3=확실한구름
        # 이진화: 0(맑음), 1(구름)
        binary_mask = np.where(cloud_mask >= 2, 1, 0)
        
        # 결측값(-1) 처리
        binary_mask = np.where(cloud_mask == -1, -1, binary_mask)
        
        return binary_mask.astype(np.int8)
    
    def create_image_patches(self, cloud_mask, patch_size=(256, 256), overlap=0.5):
        """이미지를 패치로 분할"""
        h, w = cloud_mask.shape
        step = int(patch_size[0] * (1 - overlap))
        
        patches = []
        positions = []
        
        for i in range(0, h - patch_size[0] + 1, step):
            for j in range(0, w - patch_size[1] + 1, step):
                patch = cloud_mask[i:i+patch_size[0], j:j+patch_size[1]]
                patches.append(patch)
                positions.append((i, j))
        
        return np.array(patches), positions
    
    def visualize_cloud_data(self, cloud_mask, title="GK2A Cloud Mask", save_path=None):
        """구름 데이터 시각화"""
        plt.figure(figsize=(12, 8))
        
        # 색상 맵 정의: 흰색(맑음), 파란색(구름), 회색(결측)
        colors = ['white', 'lightblue', 'blue', 'darkblue', 'gray']
        cmap = ListedColormap(colors[:np.max(cloud_mask)+2])
        
        plt.imshow(cloud_mask, cmap=cmap, vmin=-1, vmax=3)
        plt.colorbar(label='Cloud Mask (0:Clear, 1-3:Cloud levels, -1:Missing)')
        plt.title(title)
        plt.xlabel('Longitude Index')
        plt.ylabel('Latitude Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 이미지 저장: {save_path}")
        
        plt.show()
    
    def collect_time_series(self, start_date, end_date, interval_minutes=10):
        """시계열 데이터 수집을 위한 파일명 리스트 생성"""
        file_list = []
        current = datetime.strptime(start_date, '%Y%m%d%H%M')
        end = datetime.strptime(end_date, '%Y%m%d%H%M')
        
        while current <= end:
            date_str = current.strftime('%Y%m%d%H%M')
            filename = f"gk2a_ami_le2_cld_ko020lc_{date_str}.nc"
            filepath = os.path.join(self.data_folder, filename)
            
            if os.path.exists(filepath):
                file_list.append(filepath)
            
            current += timedelta(minutes=interval_minutes)
        
        return file_list

# 사용 예시
if __name__ == "__main__":
    processor = GK2ACloudProcessor()
    
    # 샘플 파일 처리
    sample_file = "./gk2a_data/gk2a_ami_le2_cld_ko020lc_202409240600.nc"
    
    if os.path.exists(sample_file):
        # 데이터 로드
        cloud_mask, metadata = processor.load_cloud_data(sample_file)
        
        if cloud_mask is not None:
            print(f"✅ 데이터 로드 성공: {cloud_mask.shape}")
            print(f"📊 메타데이터: {metadata}")
            
            # 전처리
            binary_mask = processor.preprocess_cloud_mask(cloud_mask)
            
            # 시각화
            processor.visualize_cloud_data(binary_mask, 
                                         title=f"GK2A Cloud Mask - {metadata['time']}")
            
            # 패치 분할
            patches, positions = processor.create_image_patches(binary_mask)
            print(f"✅ 패치 생성 완료: {len(patches)}개 패치")
    else:
        print(f"❌ 샘플 파일이 없습니다: {sample_file}")
        print("먼저 gk2a_downloader.py를 실행하여 데이터를 다운로드하세요.")
'''

# 전처리 코드 저장
with open('gk2a_preprocessor.py', 'w', encoding='utf-8') as f:
    f.write(preprocessing_code)

print("✅ GK2A 데이터 전처리 코드 생성 완료: gk2a_preprocessor.py")
print("\n📦 필요한 패키지 설치:")
print("pip install xarray netcdf4 matplotlib numpy pandas")
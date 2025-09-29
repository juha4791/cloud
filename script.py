# GK2A 데이터 접근을 위한 단계별 체크리스트 생성
import pandas as pd

checklist_data = {
    "단계": ["API 키 발급", "데이터 접근 테스트", "저장소 설정", "전처리 파이프라인"],
    "필요 작업": [
        "기상청 API 허브 회원가입 → apihub.kma.go.kr",
        "GK2A 구름탐지(CLD) 데이터 다운로드 테스트", 
        "대용량 데이터 저장을 위한 로컬/클라우드 스토리지",
        "NetCDF 파일 읽기 및 이미지 변환 코드"
    ],
    "예상 소요시간": ["1일", "2-3일", "1일", "3-5일"],
    "중요도": ["필수", "필수", "필수", "필수"]
}

checklist_df = pd.DataFrame(checklist_data)
print("=== 1단계: 데이터 수집 환경 구축 ===")
print(checklist_df.to_string(index=False))
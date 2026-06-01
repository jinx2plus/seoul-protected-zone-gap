# 서울 보호구역 밖 300m 분석 프로젝트

서울시 보행약자(어린이·고령자) 사고다발 지역과 보호구역 경계 밖 생활동선을 결합해, 안전시설 우선 배치 후보지를 도출하는 분석 패키지입니다.  
원천 데이터는 `data/`에, 처리 결과는 `output/processed`에, 시각화 결과는 `output/시각화결과물`에 저장됩니다.

## 폴더 구성

```
data/                            # 원천 CSV/XLSX/SHP/GPKG/ZIP
output/
  processed/                     # 중간·최종 처리 결과(GPKG/CSV)
  jupyter-notebook/              # 분석 노트북
  시각화결과물/                   # HTML 지도/대시보드
scripts/
  build_protected_zone_notebook.py
  build_visual_outputs.py
```

## 실행 순서

1. **환경 준비**
   - Python 3.11
   - 필수 패키지: `geopandas`, `pandas`, `numpy`, `shapely`, `folium`, `matplotlib`, `branca`

2. **데이터 가공 노트북 생성**
   ```powershell
   python .\scripts\build_protected_zone_notebook.py
   ```
   생성된 `output/jupyter-notebook/protected_zone_300m_data_processing.ipynb`를 열어 셀을 순서대로 실행합니다.

3. **시각화 산출물 생성**
   ```powershell
   python .\scripts\build_visual_outputs.py
   ```
   `output/시각화결과물`에 HTML 지도/요약 대시보드와 이미지가 생성됩니다.

> PPTX 생성 코드는 제거되어 있습니다.

## 환경변수 (선택)

데이터 위치가 바뀐 경우 아래 환경변수로 경로를 지정할 수 있습니다.

```powershell
setx SEOUL_PROJECT_ROOT "D:\new\root"
setx SEOUL_DATA_DIR "D:\new\root\data"
setx SEOUL_OUTPUT_DIR "D:\new\root\output"
setx SEOUL_PROCESSED_DIR "D:\new\root\output\processed"
```

## 주요 산출물

- `output/processed/seoul_accident_priority_top30.csv`
- `output/processed/seoul_accident_priority_areas.gpkg`
- `output/processed/seoul_admin_safety_scores.gpkg`
- `output/processed/seoul_processed_facility_layers.gpkg`
- `output/시각화결과물/*.html`

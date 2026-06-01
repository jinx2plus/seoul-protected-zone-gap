import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SEOUL_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
NB_PATH = ROOT / "output" / "jupyter-notebook" / "protected_zone_300m_data_processing.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(keepends=True)}


cells = [
    md(
        """
# 보호구역 밖 300m 데이터 가공 노트북

`data/` 폴더의 원천데이터를 계획서 기준으로 정리하여 서울 어린이·고령자 보행안전 사각지대와 안전시설 우선배치 후보지를 산출한다.

주요 산출물:
- `output/processed/seoul_accident_priority_top30.csv`
- `output/processed/seoul_accident_priority_areas.gpkg`
- `output/processed/seoul_admin_safety_scores.gpkg`
- `output/processed/processing_manifest.csv`

거리 기반 계산은 모두 `EPSG:32652`에서 수행한다.
"""
    ),
    md(
        """
## 0. 처리 원칙

- WGS84 원천 좌표는 `EPSG:4326`으로 선언한 뒤 `EPSG:32652`로 변환한다.
- 서울시 인허가 `좌표정보(X)`, `좌표정보(Y)`는 `EPSG:5174`로 처리한다.
- 초등학교 통학구역 SHP는 `EPSG:5186`로 읽은 뒤 변환한다.
- TAAS 사고다발지역 polygon 좌표는 `EPSG:5179`로 해석하고, 실패하면 중심 위경도를 사용한다.
- 보호구역 polygon이 없고 중심점만 있는 경우 100m/300m/600m/1000m 링은 “보호구역 중심 기준”이다.
"""
    ),
    code(
        """
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, shape
from shapely.ops import unary_union

warnings.filterwarnings("ignore", category=UserWarning)

def resolve_project_root() -> Path:
    env_root = os.environ.get("SEOUL_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (base / "data").exists() and (base / "output").exists():
            return base
    return cwd

PROJECT_ROOT = resolve_project_root()
DATA_DIR = Path(os.environ.get("SEOUL_DATA_DIR", PROJECT_ROOT / "data")).resolve()
OUT_DIR = Path(os.environ.get("SEOUL_OUTPUT_DIR", PROJECT_ROOT / "output" / "processed")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("PROJECT_ROOT =", PROJECT_ROOT)

TARGET_CRS = "EPSG:32652"
WGS84 = "EPSG:4326"
LOCALDATA_CRS = "EPSG:5174"
TAAS_POLYGON_CRS = "EPSG:5179"

SEOUL_BBOX = {"min_lon": 126.70, "max_lon": 127.25, "min_lat": 37.40, "max_lat": 37.75}

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
"""
    ),
    md(
        """
## 1. 공통 유틸리티

파일명은 다운로드 시점마다 달라질 수 있으므로 glob 패턴으로 찾는다. 좌표 컬럼은 숫자 변환, 서울 필터, 좌표 유효성 검사를 공통 처리한다.
"""
    ),
    code(
        """
def first_existing(patterns: str | Iterable[str], required: bool = False) -> Path | None:
    if isinstance(patterns, str):
        patterns = [patterns]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(DATA_DIR.glob(pattern))
    matches = sorted(set(matches), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"No file matched: {patterns}")
    return None


def read_csv_smart(path: Path, **kwargs) -> pd.DataFrame:
    errors = []
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except Exception as exc:
            errors.append((enc, repr(exc)))
    raise RuntimeError(f"CSV read failed: {path}\\n{errors}")


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return read_csv_smart(path, **kwargs)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported table type: {path}")


def num(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(text.replace({"": np.nan, "nan": np.nan, "None": np.nan}), errors="coerce")


def filter_seoul_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = pd.Series(False, index=out.index)
    text_cols = [
        "시도명", "사고다발지역시도시군구", "소재지도로명주소", "소재지지번주소",
        "도로명주소", "지번주소", "역사도로명주소", "관리기관명", "제공기관명",
    ]
    for col in text_cols:
        if col in out.columns:
            mask |= out[col].astype(str).str.contains("서울", na=False)
    return out[mask].copy() if mask.any() else out


def valid_lonlat_mask(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.Series:
    lon = num(df[lon_col])
    lat = num(df[lat_col])
    return (
        lon.between(SEOUL_BBOX["min_lon"], SEOUL_BBOX["max_lon"])
        & lat.between(SEOUL_BBOX["min_lat"], SEOUL_BBOX["max_lat"])
    )


def points_from_lonlat(df: pd.DataFrame, lon_col: str, lat_col: str, source_name: str) -> gpd.GeoDataFrame:
    work = df.copy()
    if work.empty:
        return gpd.GeoDataFrame(work, geometry=[], crs=TARGET_CRS)
    work[lon_col] = num(work[lon_col])
    work[lat_col] = num(work[lat_col])
    work = work[valid_lonlat_mask(work, lon_col, lat_col)].copy()
    if work.empty:
        return gpd.GeoDataFrame(work, geometry=[], crs=TARGET_CRS)
    gdf = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work[lon_col], work[lat_col]), crs=WGS84).to_crs(TARGET_CRS)
    gdf["source_layer"] = source_name
    return gdf


def points_from_xy(df: pd.DataFrame, x_col: str, y_col: str, source_crs: str, source_name: str) -> gpd.GeoDataFrame:
    work = df.copy()
    if work.empty:
        return gpd.GeoDataFrame(work, geometry=[], crs=TARGET_CRS)
    work[x_col] = num(work[x_col])
    work[y_col] = num(work[y_col])
    work = work.dropna(subset=[x_col, y_col]).copy()
    if work.empty:
        return gpd.GeoDataFrame(work, geometry=[], crs=TARGET_CRS)
    gdf = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work[x_col], work[y_col]), crs=source_crs).to_crs(TARGET_CRS)
    gdf["source_layer"] = source_name
    return gdf


def lines_from_lonlat(df: pd.DataFrame, start_lon: str, start_lat: str, end_lon: str, end_lat: str, source_name: str) -> gpd.GeoDataFrame:
    work = df.copy()
    if work.empty:
        return gpd.GeoDataFrame(work, geometry=[], crs=TARGET_CRS)
    for col in [start_lon, start_lat, end_lon, end_lat]:
        work[col] = num(work[col])
    work = work.dropna(subset=[start_lon, start_lat, end_lon, end_lat]).copy()
    if work.empty:
        return gpd.GeoDataFrame(work, geometry=[], crs=TARGET_CRS)
    work["geometry"] = [LineString([(a, b), (c, d)]) for a, b, c, d in zip(work[start_lon], work[start_lat], work[end_lon], work[end_lat])]
    gdf = gpd.GeoDataFrame(work, geometry="geometry", crs=WGS84).to_crs(TARGET_CRS)
    gdf["source_layer"] = source_name
    return gdf


def clip_to_seoul(gdf: gpd.GeoDataFrame, seoul_geom) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    work = gdf.to_crs(TARGET_CRS)
    return work[work.geometry.notna() & work.geometry.intersects(seoul_geom)].copy()


def minmax_0_100(s: pd.Series) -> pd.Series:
    values = num(s).fillna(0)
    mn, mx = values.min(), values.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or math.isclose(mx, mn):
        return pd.Series(0.0, index=values.index)
    return ((values - mn) / (mx - mn) * 100).clip(0, 100)


def active_only(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["영업상태명", "상세영업상태명"]:
        if col in df.columns:
            return df[df[col].astype(str).str.contains("영업|정상|운영", na=False)].copy()
    return df


def safe_columns(gdf: gpd.GeoDataFrame, max_text_len: int = 500) -> gpd.GeoDataFrame:
    work = gdf.copy()
    for col in work.columns:
        if col == work.geometry.name:
            continue
        if pd.api.types.is_object_dtype(work[col]):
            work[col] = work[col].astype(str).str.slice(0, max_text_len).replace({"nan": None, "None": None})
    return work
"""
    ),
    md(
        """
## 2. 원천 파일 탐색

계획서의 데이터군과 현재 `data/` 폴더 파일을 매핑한다. 누락 파일은 manifest에 기록하고 해당 레이어만 건너뛴다.
"""
    ),
    code(
        """
FILE_PATTERNS = {
    "admin": ["행정동*.gpkg"],
    "accident": ["*교통사고다발지역*.csv"],
    "child_zone": ["*어린이보호구역*.csv"],
    "senior_zone": ["*노인장애인보호구역*.csv", "*노인*장애인*보호구역*.csv"],
    "crosswalk": ["*횡단보도*.csv"],
    "signal": ["*신호등*.csv"],
    "speed_bump": ["과속방지턱정보_서울특별시.csv", "*과속방지턱*.csv"],
    "enforcement_camera": ["*무인교통단속카메라*.csv"],
    "road_sign": ["*도로안전표지*.csv"],
    "security_light": ["*보안등*.csv"],
    "ped_priority": ["*보행자우선도로*.csv"],
    "ped_only": ["*보행자전용도로*.csv"],
    "oneway": ["*일방통행도로*.csv"],
    "cctv": ["CCTV정보_서울특별시.csv", "*CCTV*.csv"],
    "senior_center": ["*마을회관및경로당*.csv"],
    "hospital": ["건강_병원.csv"],
    "clinic": ["건강_의원.csv", "건강_부속의료기관.csv"],
    "pharmacy": ["건강_약국.csv"],
    "bus_stop": ["서울시버스정류소위치정보*.xlsx"],
    "subway_station": ["전체_도시철도역사정보*.xlsx"],
    "school_district": ["초등학교통학구역.shp"],
    "age_population": ["*연령별인구현황*.csv"],
    "total_population": ["주민등록 세대 및 인구*.csv"],
}

manifest = []
paths: dict[str, Path | None] = {}
for key, patterns in FILE_PATTERNS.items():
    path = first_existing(patterns)
    paths[key] = path
    manifest.append({"layer": key, "path": str(path) if path else None, "exists": path is not None})

manifest_df = pd.DataFrame(manifest)
display(manifest_df)
manifest_df.to_csv(OUT_DIR / "processing_manifest.csv", index=False, encoding="utf-8-sig")
"""
    ),
    md("## 3. 서울 경계와 행정동"),
    code(
        """
admin_path = paths["admin"]
if admin_path is None:
    raise FileNotFoundError("행정동 경계 GPKG가 필요합니다. data/행정동*.gpkg 파일을 넣어주세요.")

admin = gpd.read_file(admin_path).to_crs(TARGET_CRS)
admin["ADSTRD_CD"] = admin["ADSTRD_CD"].astype(str).str[:8]
admin["admin_area_m2"] = admin.geometry.area
seoul_geom = unary_union(admin.geometry)

print(admin_path.name, admin.crs, len(admin))
display(admin[["ADSTRD_CD", "ADSTRD_NM", "admin_area_m2"]].head())
"""
    ),
    md("## 4. 사고다발지역 처리"),
    code(
        """
def parse_taas_polygon(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return shape(json.loads(text))
    except Exception:
        return None


def load_accidents(path: Path | None) -> gpd.GeoDataFrame:
    if path is None:
        return gpd.GeoDataFrame(columns=["accident_id", "geometry"], geometry="geometry", crs=TARGET_CRS)
    df = filter_seoul_rows(read_csv_smart(path))
    if "사고유형구분" in df.columns:
        df = df[df["사고유형구분"].astype(str).str.contains("스쿨존어린이|보행어린이|보행노인|무단횡단", na=False)].copy()
    point_gdf = points_from_lonlat(df, "경도", "위도", "accident_center")
    if point_gdf.empty:
        return point_gdf
    poly_series = point_gdf["사고다발지역폴리곤정보"].map(parse_taas_polygon) if "사고다발지역폴리곤정보" in point_gdf.columns else pd.Series([None] * len(point_gdf), index=point_gdf.index)
    valid_poly = poly_series.notna()
    if valid_poly.any():
        poly_gdf = gpd.GeoDataFrame(point_gdf.drop(columns="geometry"), geometry=poly_series, crs=TAAS_POLYGON_CRS).to_crs(TARGET_CRS)
        point_gdf.loc[valid_poly, "geometry"] = poly_gdf.loc[valid_poly, "geometry"]
        point_gdf["geometry_source"] = np.where(valid_poly, "taas_polygon", "lonlat_center")
    else:
        point_gdf["geometry_source"] = "lonlat_center"
    point_gdf = clip_to_seoul(point_gdf, seoul_geom).reset_index(drop=True)
    point_gdf["accident_id"] = [f"ACC_{i:05d}" for i in range(len(point_gdf))]
    return point_gdf


accidents = load_accidents(paths["accident"])
print("accidents", len(accidents))
if not accidents.empty:
    display(accidents[["accident_id", "사고연도", "사고유형구분", "사고지역위치명", "사고건수", "사망자수", "중상자수", "경상자수", "부상신고자수", "geometry_source"]].head())
"""
    ),
    md("## 5. 보호구역, 안전시설, 생활시설 레이어 생성"),
    code(
        """
def empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=["geometry", "source_layer"], geometry="geometry", crs=TARGET_CRS)


def load_lonlat_layer(key: str, lon_col: str = "경도", lat_col: str = "위도") -> gpd.GeoDataFrame:
    path = paths.get(key)
    if path is None:
        return empty_gdf()
    df = filter_seoul_rows(read_table(path))
    if lon_col not in df.columns or lat_col not in df.columns:
        return empty_gdf()
    return clip_to_seoul(points_from_lonlat(df, lon_col, lat_col, key), seoul_geom)


def load_localdata_xy_layer(key: str) -> gpd.GeoDataFrame:
    path = paths.get(key)
    if path is None:
        return empty_gdf()
    df = filter_seoul_rows(active_only(read_csv_smart(path)))
    if "좌표정보(X)" not in df.columns or "좌표정보(Y)" not in df.columns:
        return empty_gdf()
    return clip_to_seoul(points_from_xy(df, "좌표정보(X)", "좌표정보(Y)", LOCALDATA_CRS, key), seoul_geom)


def load_bus_stop() -> gpd.GeoDataFrame:
    path = paths.get("bus_stop")
    return empty_gdf() if path is None else clip_to_seoul(points_from_lonlat(pd.read_excel(path), "X좌표", "Y좌표", "bus_stop"), seoul_geom)


def load_subway_station() -> gpd.GeoDataFrame:
    path = paths.get("subway_station")
    if path is None:
        return empty_gdf()
    return clip_to_seoul(points_from_lonlat(filter_seoul_rows(pd.read_excel(path)), "역경도", "역위도", "subway_station"), seoul_geom)


def load_school_district() -> gpd.GeoDataFrame:
    path = paths.get("school_district")
    if path is None:
        return empty_gdf()
    gdf = gpd.read_file(path).to_crs(TARGET_CRS)
    gdf["source_layer"] = "school_district"
    return clip_to_seoul(gdf, seoul_geom)


def load_line_layer(key: str, start_lat: str, start_lon: str, end_lat: str, end_lon: str) -> gpd.GeoDataFrame:
    path = paths.get(key)
    if path is None:
        return empty_gdf()
    df = filter_seoul_rows(read_csv_smart(path))
    if any(c not in df.columns for c in [start_lat, start_lon, end_lat, end_lon]):
        return empty_gdf()
    return clip_to_seoul(lines_from_lonlat(df, start_lon, start_lat, end_lon, end_lat, key), seoul_geom)


layers = {
    "child_zone": load_lonlat_layer("child_zone"),
    "senior_zone": load_lonlat_layer("senior_zone"),
    "crosswalk": load_lonlat_layer("crosswalk"),
    "signal": load_lonlat_layer("signal"),
    "speed_bump": load_lonlat_layer("speed_bump", "WGS84경도", "WGS84위도"),
    "enforcement_camera": load_lonlat_layer("enforcement_camera"),
    "road_sign": load_lonlat_layer("road_sign"),
    "security_light": load_lonlat_layer("security_light"),
    "cctv": load_lonlat_layer("cctv", "WGS84경도", "WGS84위도"),
    "senior_center": load_lonlat_layer("senior_center"),
    "hospital": load_localdata_xy_layer("hospital"),
    "clinic": load_localdata_xy_layer("clinic"),
    "pharmacy": load_localdata_xy_layer("pharmacy"),
    "bus_stop": load_bus_stop(),
    "subway_station": load_subway_station(),
    "school_district": load_school_district(),
    "ped_priority": load_line_layer("ped_priority", "보행자우선도로시작점위도", "보행자우선도로시작점경도", "보행자우선도로종료점위도", "보행자우선도로종료점경도"),
    "ped_only": load_line_layer("ped_only", "보행자전용도로시작점위도", "보행자전용도로시작점경도", "보행자전용도로종료점위도", "보행자전용도로종료점경도"),
    "oneway": load_line_layer("oneway", "시작점위도", "시작점경도", "종료점위도", "종료점경도"),
}

layer_summary = pd.DataFrame([{"layer": k, "rows": len(v), "geom_type": ",".join(sorted(v.geom_type.dropna().unique())) if not v.empty else ""} for k, v in layers.items()])
display(layer_summary.sort_values("rows", ascending=False))
"""
    ),
    md("## 6. 인구 노출도 처리"),
    code(
        """
def load_age_population() -> pd.DataFrame:
    path = paths.get("age_population")
    if path is None:
        return pd.DataFrame(columns=["ADSTRD_CD", "child_pop_proxy", "senior_pop_proxy"])
    work = read_csv_smart(path)
    if "행정구역" not in work.columns:
        return pd.DataFrame(columns=["ADSTRD_CD", "child_pop_proxy", "senior_pop_proxy"])
    work["ADSTRD_CD"] = work["행정구역"].astype(str).str.extract(r"\\((\\d+)\\)")[0].str[:8]
    work = work[work["ADSTRD_CD"].notna()].copy()

    def pick(token: str) -> pd.Series:
        cols = [c for c in work.columns if token in c and not c.startswith(("2026년04월_남", "2026년04월_여"))]
        return num(work[cols[0]]) if cols else pd.Series(0, index=work.index, dtype="float64")

    work["child_pop_proxy"] = pick("거주자_0~9세") + 0.4 * pick("거주자_10~19세")
    work["senior_pop_proxy"] = 0.5 * pick("거주자_60~69세") + pick("거주자_70~79세") + pick("거주자_80~89세") + pick("거주자_90~99세") + pick("거주자_100세 이상")
    return work[["ADSTRD_CD", "child_pop_proxy", "senior_pop_proxy"]].drop_duplicates("ADSTRD_CD")


pop = load_age_population()
admin_pop = admin.merge(pop, on="ADSTRD_CD", how="left")
admin_pop[["child_pop_proxy", "senior_pop_proxy"]] = admin_pop[["child_pop_proxy", "senior_pop_proxy"]].fillna(0)
display(admin_pop[["ADSTRD_CD", "ADSTRD_NM", "child_pop_proxy", "senior_pop_proxy"]].head())
"""
    ),
    md("## 7. 보호구역 링과 사고권역 보호공백 산정"),
    code(
        """
def nearest_distance(base_centers: gpd.GeoDataFrame, target: gpd.GeoDataFrame, distance_col: str) -> pd.Series:
    if base_centers.empty or target is None or target.empty:
        return pd.Series(np.nan, index=base_centers.index)
    nearest = gpd.sjoin_nearest(base_centers[["accident_id", "geometry"]], target[["geometry"]], how="left", distance_col=distance_col)
    return nearest.groupby("accident_id")[distance_col].min().reindex(base_centers["accident_id"]).reset_index(drop=True)


def gap_score_from_distance(dist: pd.Series) -> pd.Series:
    return pd.cut(dist.fillna(np.inf), bins=[-np.inf, 100, 300, 600, np.inf], labels=[0, 30, 70, 100]).astype(float)


def ring_label_from_distance(dist: pd.Series) -> pd.Series:
    return pd.cut(
        dist.fillna(np.inf),
        bins=[-np.inf, 100, 300, 600, 1000, np.inf],
        labels=["inside_0_100m", "edge_100_300m", "outside_300_600m", "extended_600_1000m", "beyond_1000m"],
    ).astype(str)


accident_centers = accidents.copy()
if not accident_centers.empty:
    accident_centers["geometry"] = accident_centers.geometry.representative_point()
    child_dist = nearest_distance(accident_centers, layers["child_zone"], "child_zone_dist_m")
    senior_dist = nearest_distance(accident_centers, layers["senior_zone"], "senior_zone_dist_m")
    is_senior = accidents["사고유형구분"].astype(str).str.contains("노인", na=False).reset_index(drop=True)
    selected_dist = child_dist.where(~is_senior, senior_dist)
    accidents["child_zone_dist_m"] = child_dist.values
    accidents["senior_zone_dist_m"] = senior_dist.values
    accidents["protection_dist_m"] = selected_dist.values
    accidents["protection_ring"] = ring_label_from_distance(selected_dist).values
    accidents["protection_gap_score"] = gap_score_from_distance(selected_dist).values
    display(accidents[["accident_id", "사고유형구분", "protection_dist_m", "protection_ring", "protection_gap_score"]].head())
"""
    ),
    md("## 8. 사고권역 주변 시설 공급량 집계"),
    code(
        """
def count_features_within(base_centers: gpd.GeoDataFrame, features: gpd.GeoDataFrame, radius_m: float, out_col: str, predicate: str = "within") -> pd.Series:
    if base_centers.empty or features is None or features.empty:
        return pd.Series(0, index=base_centers.index, name=out_col, dtype="int64")
    buffers = base_centers[["accident_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius_m)
    joined = gpd.sjoin(features[["geometry"]], buffers, how="inner", predicate=predicate)
    counts = joined.groupby("accident_id").size()
    return base_centers["accident_id"].map(counts).fillna(0).astype("int64").rename(out_col)


facility_count_specs = [
    ("crosswalk", 200, "crosswalk_200m", "within"),
    ("signal", 200, "signal_200m", "within"),
    ("speed_bump", 200, "speed_bump_200m", "within"),
    ("enforcement_camera", 300, "enforcement_camera_300m", "within"),
    ("cctv", 200, "cctv_200m", "within"),
    ("road_sign", 200, "road_sign_200m", "within"),
    ("security_light", 200, "security_light_200m", "within"),
    ("ped_priority", 300, "ped_priority_300m", "intersects"),
    ("ped_only", 300, "ped_only_300m", "intersects"),
    ("oneway", 300, "oneway_300m", "intersects"),
    ("bus_stop", 300, "bus_stop_300m", "within"),
    ("subway_station", 600, "subway_station_600m", "within"),
    ("senior_center", 600, "senior_center_600m", "within"),
    ("hospital", 600, "hospital_600m", "within"),
    ("clinic", 600, "clinic_600m", "within"),
    ("pharmacy", 600, "pharmacy_600m", "within"),
    ("child_zone", 600, "child_zone_600m", "within"),
    ("senior_zone", 600, "senior_zone_600m", "within"),
    ("school_district", 300, "school_district_300m", "intersects"),
]

if not accident_centers.empty:
    count_df = pd.DataFrame({"accident_id": accident_centers["accident_id"].values})
    for layer_name, radius, out_col, predicate in facility_count_specs:
        count_df[out_col] = count_features_within(accident_centers, layers[layer_name], radius, out_col, predicate).values
    accidents = accidents.merge(count_df, on="accident_id", how="left")
    display(count_df.head())
"""
    ),
    md("## 9. 행정동 매칭과 지수 산정"),
    code(
        """
def attach_admin(base_centers: gpd.GeoDataFrame, admin_layer: gpd.GeoDataFrame) -> pd.DataFrame:
    if base_centers.empty:
        return pd.DataFrame(columns=["accident_id", "ADSTRD_CD", "ADSTRD_NM"])
    joined = gpd.sjoin(base_centers[["accident_id", "geometry"]], admin_layer[["ADSTRD_CD", "ADSTRD_NM", "geometry"]], how="left", predicate="within")
    return joined[["accident_id", "ADSTRD_CD", "ADSTRD_NM"]].drop_duplicates("accident_id")


if not accidents.empty:
    admin_match = attach_admin(accident_centers, admin_pop)
    accidents = accidents.merge(admin_match, on="accident_id", how="left")
    accidents = accidents.merge(admin_pop[["ADSTRD_CD", "child_pop_proxy", "senior_pop_proxy"]], on="ADSTRD_CD", how="left")

    for col in ["사고건수", "사망자수", "중상자수", "경상자수", "부상신고자수"]:
        accidents[col] = num(accidents[col]) if col in accidents.columns else 0

    accidents["severity_raw"] = (
        minmax_0_100(accidents["사고건수"])
        + accidents["사망자수"].fillna(0) * 5
        + accidents["중상자수"].fillna(0) * 3
        + accidents["경상자수"].fillna(0)
        + accidents["부상신고자수"].fillna(0) * 0.5
    )
    accidents["accident_severity_score"] = minmax_0_100(accidents["severity_raw"])

    count_cols = [c for c in accidents.columns if c.endswith(("_200m", "_300m", "_600m"))]
    accidents[count_cols] = accidents[count_cols].fillna(0)
    accidents["safety_supply_raw"] = (
        accidents.get("crosswalk_200m", 0) * 2.0
        + accidents.get("signal_200m", 0) * 2.0
        + accidents.get("speed_bump_200m", 0) * 1.5
        + accidents.get("enforcement_camera_300m", 0) * 1.5
        + accidents.get("cctv_200m", 0)
        + accidents.get("road_sign_200m", 0) * 0.5
        + accidents.get("security_light_200m", 0) * 0.5
        + accidents.get("ped_priority_300m", 0) * 2.0
        + accidents.get("ped_only_300m", 0) * 1.5
    )
    accidents["safety_supply_score"] = minmax_0_100(accidents["safety_supply_raw"])
    accidents["safety_shortage_score"] = 100 - accidents["safety_supply_score"]
    accidents["child_life_raw"] = accidents.get("child_zone_600m", 0) + accidents.get("school_district_300m", 0)
    accidents["senior_life_raw"] = accidents.get("senior_zone_600m", 0) + accidents.get("senior_center_600m", 0) + accidents.get("hospital_600m", 0) + accidents.get("clinic_600m", 0) + accidents.get("pharmacy_600m", 0)
    accidents["transit_risk_raw"] = accidents.get("bus_stop_300m", 0) + accidents.get("subway_station_600m", 0)
    accidents["child_life_score"] = minmax_0_100(accidents["child_life_raw"])
    accidents["senior_life_score"] = minmax_0_100(accidents["senior_life_raw"])
    accidents["transit_risk_score"] = minmax_0_100(accidents["transit_risk_raw"])
    accidents["child_pop_score"] = minmax_0_100(accidents["child_pop_proxy"].fillna(0))
    accidents["senior_pop_score"] = minmax_0_100(accidents["senior_pop_proxy"].fillna(0))
    accidents["child_blindspot_index"] = accidents["accident_severity_score"] * 0.35 + accidents["protection_gap_score"].fillna(100) * 0.25 + accidents["child_life_score"] * 0.20 + accidents["safety_shortage_score"] * 0.15 + accidents["child_pop_score"] * 0.05
    accidents["senior_blindspot_index"] = accidents["accident_severity_score"] * 0.35 + accidents["senior_life_score"] * 0.20 + accidents["safety_shortage_score"] * 0.20 + accidents["senior_pop_score"] * 0.15 + accidents["transit_risk_score"] * 0.10
    accidents["complex_risk_index"] = np.minimum(accidents["child_blindspot_index"], accidents["senior_blindspot_index"])
    accidents["intervention_priority_index"] = (np.maximum(accidents["child_blindspot_index"], accidents["senior_blindspot_index"]) + accidents["complex_risk_index"] * 0.15).clip(0, 100)
    accidents["priority_grade"] = pd.cut(accidents["intervention_priority_index"], bins=[-np.inf, 40, 60, 80, np.inf], labels=["C_observe", "B_facility_supplement", "A_preventive_management", "S_immediate_improvement"]).astype(str)
    display(accidents[["accident_id", "ADSTRD_NM", "사고유형구분", "protection_ring", "accident_severity_score", "safety_shortage_score", "child_blindspot_index", "senior_blindspot_index", "intervention_priority_index", "priority_grade"]].head())
"""
    ),
    md("## 10. 부족시설과 정책 패키지 자동 라벨링"),
    code(
        """
def missing_facilities(row: pd.Series) -> str:
    missing = []
    thresholds = {
        "crosswalk_200m": (1, "횡단보도"),
        "signal_200m": (1, "신호등"),
        "speed_bump_200m": (1, "과속방지턱"),
        "enforcement_camera_300m": (1, "무인단속카메라"),
        "cctv_200m": (1, "CCTV"),
        "security_light_200m": (1, "보안등"),
        "ped_priority_300m": (1, "보행자우선도로"),
    }
    for col, (threshold, label) in thresholds.items():
        if row.get(col, 0) < threshold:
            missing.append(label)
    return ", ".join(missing) if missing else "상대적으로 충분"


def policy_package(row: pd.Series) -> str:
    accident_type = str(row.get("사고유형구분", ""))
    missing = str(row.get("missing_facilities", ""))
    if "노인" in accident_type and (row.get("hospital_600m", 0) + row.get("clinic_600m", 0) + row.get("pharmacy_600m", 0)) > 0:
        return "고령자 병원형: 보행신호 연장, 횡단거리 단축, 보행섬 검토"
    if "노인" in accident_type and row.get("bus_stop_300m", 0) > 0:
        return "정류장 접근형: 정류장-횡단보도 동선 정비"
    if "어린이" in accident_type and row.get("protection_ring") in ["outside_300_600m", "extended_600_1000m", "beyond_1000m"]:
        return "어린이 보호구역 밖 300m형: 보호구역 확장, 속도저감, 단속 강화"
    if "과속방지턱" in missing or "무인단속카메라" in missing:
        return "속도저감형: 과속방지턱, 단속카메라, 제한속도 관리"
    if "보안등" in missing or "CCTV" in missing:
        return "야간취약형: 보안등, CCTV, 안전비상벨 보강"
    if "횡단보도" in missing or "신호등" in missing:
        return "횡단시설형: 횡단보도, 보행신호, 음향신호기 보강"
    return "복합위험형: 현장점검 후 보행안전 특별관리구역 검토"


if not accidents.empty:
    accidents["missing_facilities"] = accidents.apply(missing_facilities, axis=1)
    accidents["policy_package"] = accidents.apply(policy_package, axis=1)
    top30 = accidents.sort_values("intervention_priority_index", ascending=False).head(30).copy()
    top30["rank"] = np.arange(1, len(top30) + 1)
    display(top30[["rank", "ADSTRD_NM", "사고유형구분", "protection_ring", "missing_facilities", "policy_package", "intervention_priority_index", "priority_grade"]])
else:
    top30 = accidents.copy()
"""
    ),
    md("## 11. 행정동별 점수 집계"),
    code(
        """
if not accidents.empty:
    admin_scores = accidents.groupby(["ADSTRD_CD", "ADSTRD_NM"], dropna=False).agg(
        accident_area_count=("accident_id", "count"),
        mean_priority=("intervention_priority_index", "mean"),
        max_priority=("intervention_priority_index", "max"),
        mean_safety_shortage=("safety_shortage_score", "mean"),
        s_grade_count=("priority_grade", lambda s: (s == "S_immediate_improvement").sum()),
        a_grade_count=("priority_grade", lambda s: (s == "A_preventive_management").sum()),
    ).reset_index()
    admin_result = admin_pop.merge(admin_scores, on=["ADSTRD_CD", "ADSTRD_NM"], how="left")
else:
    admin_result = admin_pop.copy()

for col in ["accident_area_count", "mean_priority", "max_priority", "mean_safety_shortage", "s_grade_count", "a_grade_count"]:
    admin_result[col] = admin_result.get(col, 0).fillna(0)

display(admin_result[["ADSTRD_CD", "ADSTRD_NM", "accident_area_count", "mean_priority", "max_priority", "s_grade_count", "a_grade_count"]].sort_values("max_priority", ascending=False).head(20))
"""
    ),
    md("## 12. 결과 저장"),
    code(
        """
if not accidents.empty:
    csv_cols = [
        "rank", "accident_id", "ADSTRD_CD", "ADSTRD_NM", "사고연도", "사고유형구분", "사고지역위치명",
        "사고건수", "사망자수", "중상자수", "경상자수", "부상신고자수",
        "protection_ring", "protection_dist_m", "missing_facilities", "policy_package",
        "accident_severity_score", "protection_gap_score", "safety_shortage_score",
        "child_blindspot_index", "senior_blindspot_index", "complex_risk_index", "intervention_priority_index", "priority_grade",
    ]
    top30[[c for c in csv_cols if c in top30.columns]].to_csv(OUT_DIR / "seoul_accident_priority_top30.csv", index=False, encoding="utf-8-sig")
    safe_columns(accidents).to_file(OUT_DIR / "seoul_accident_priority_areas.gpkg", layer="accident_priority", driver="GPKG")
    safe_columns(admin_result).to_file(OUT_DIR / "seoul_admin_safety_scores.gpkg", layer="admin_scores", driver="GPKG")

    facility_gpkg = OUT_DIR / "seoul_processed_facility_layers.gpkg"
    if facility_gpkg.exists():
        facility_gpkg.unlink()
    for layer_name, gdf in layers.items():
        if gdf is not None and not gdf.empty:
            keep_cols = ["source_layer", "geometry"] if "source_layer" in gdf.columns else ["geometry"]
            safe_columns(gdf[keep_cols].copy()).to_file(facility_gpkg, layer=layer_name[:50], driver="GPKG")

    print("saved:")
    print("-", OUT_DIR / "seoul_accident_priority_top30.csv")
    print("-", OUT_DIR / "seoul_accident_priority_areas.gpkg")
    print("-", OUT_DIR / "seoul_admin_safety_scores.gpkg")
    print("-", OUT_DIR / "seoul_processed_facility_layers.gpkg")
else:
    print("No accident data after filtering. Check data/전국교통사고다발지역표준데이터.csv and Seoul/type filters.")
"""
    ),
    md(
        """
## 13. 검증 체크

- 사고다발지역이 서울 내부에 남아 있는지 확인한다.
- `protection_ring`에서 `outside_300_600m` 후보가 충분히 도출되는지 확인한다.
- `missing_facilities`가 모두 “상대적으로 충분”이면 시설 반경/가중치가 너무 관대한 것이다.
- 실제 보호구역 polygon을 확보하면 7번 셀의 거리 계산을 polygon 기준으로 교체한다.
"""
    ),
    code(
        """
checks = {
    "accidents": len(accidents),
    "top30": len(top30),
    "admin_units": len(admin_result),
    "non_empty_layers": int((layer_summary["rows"] > 0).sum()),
}
print(checks)

if not accidents.empty:
    display(accidents["protection_ring"].value_counts(dropna=False).rename_axis("ring").reset_index(name="count"))
    display(accidents["missing_facilities"].value_counts().head(20).rename_axis("missing_facilities").reset_index(name="count"))
"""
    ),
]

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
nb["cells"] = cells
nb["metadata"].setdefault("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"})
nb["metadata"].setdefault("language_info", {"name": "python", "pygments_lexer": "ipython3"})
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NB_PATH.resolve())
print(len(cells), "cells")

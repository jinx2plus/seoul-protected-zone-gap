from __future__ import annotations

import base64
import html
import os
from io import BytesIO
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from branca.colormap import linear


def resolve_project_root() -> Path:
    env_root = os.environ.get("SEOUL_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (base / "data").exists() and (base / "output").exists():
            return base
    return Path(__file__).resolve().parents[1]


ROOT = resolve_project_root()
DATA_DIR = Path(os.environ.get("SEOUL_DATA_DIR", ROOT / "data")).resolve()
OUTPUT_DIR = Path(os.environ.get("SEOUL_OUTPUT_DIR", ROOT / "output")).resolve()
PROCESSED = Path(os.environ.get("SEOUL_PROCESSED_DIR", OUTPUT_DIR / "processed")).resolve()
HTML_DIR = OUTPUT_DIR / "html"
FIG_DIR = OUTPUT_DIR / "figures"
for path in [HTML_DIR, FIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_data():
    accidents = gpd.read_file(PROCESSED / "seoul_accident_priority_areas.gpkg").to_crs(4326)
    admin = gpd.read_file(PROCESSED / "seoul_admin_safety_scores.gpkg").to_crs(4326)
    top30 = pd.read_csv(PROCESSED / "seoul_accident_priority_top30.csv", encoding="utf-8-sig")
    top30_gdf = accidents.merge(top30[["accident_id", "rank"]], on="accident_id", how="inner")
    return accidents, admin, top30, top30_gdf


def value_color(value: float) -> str:
    if value >= 80:
        return "#B7192B"
    if value >= 60:
        return "#E86E2C"
    if value >= 40:
        return "#E7B645"
    return "#6A994E"


def save_priority_map(admin: gpd.GeoDataFrame, accidents: gpd.GeoDataFrame, top30: gpd.GeoDataFrame) -> Path:
    center = [37.5665, 126.9780]
    m = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron", control_scale=True)
    colormap = linear.YlOrRd_09.scale(float(admin["max_priority"].min()), float(admin["max_priority"].max()))
    colormap.caption = "행정동별 최대 개입우선순위"

    admin_simple = admin.copy()
    admin_simple["geometry"] = admin_simple.geometry.simplify(0.0002, preserve_topology=True)

    folium.GeoJson(
        admin_simple,
        name="행정동 최대 위험도",
        style_function=lambda f: {
            "fillColor": colormap(f["properties"].get("max_priority") or 0),
            "color": "#555555",
            "weight": 0.4,
            "fillOpacity": 0.72,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ADSTRD_NM", "accident_area_count", "max_priority", "mean_safety_shortage"],
            aliases=["행정동", "사고다발권역 수", "최대 우선순위", "평균 시설부족"],
            localize=True,
        ),
    ).add_to(m)

    for _, row in top30.iterrows():
        point = row.geometry.representative_point()
        popup = f"""
        <b>#{int(row['rank'])} {html.escape(str(row.get('ADSTRD_NM', '')))}</b><br>
        유형: {html.escape(str(row.get('사고유형구분', '')))}<br>
        링: {html.escape(str(row.get('protection_ring', '')))}<br>
        우선순위: {float(row.get('intervention_priority_index', 0)):.1f}<br>
        부족시설: {html.escape(str(row.get('missing_facilities', '')))}<br>
        처방: {html.escape(str(row.get('policy_package', '')))}
        """
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=max(5, 12 - int(row["rank"]) / 4),
            color="#1C1C1C",
            weight=1,
            fill=True,
            fill_color=value_color(float(row.get("intervention_priority_index", 0))),
            fill_opacity=0.9,
            tooltip=f"#{int(row['rank'])} {row.get('ADSTRD_NM', '')}",
            popup=folium.Popup(popup, max_width=420),
        ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out = HTML_DIR / "01_priority_choropleth_map.html"
    m.save(out)
    return out


def save_top30_map(top30: gpd.GeoDataFrame) -> Path:
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=12, tiles="cartodbpositron", control_scale=True)
    groups = {}
    for acc_type in sorted(top30["사고유형구분"].astype(str).unique()):
        groups[acc_type] = folium.FeatureGroup(name=acc_type, show=True).add_to(m)

    for _, row in top30.sort_values("rank").iterrows():
        point = row.geometry.representative_point()
        acc_type = str(row.get("사고유형구분", "기타"))
        color = "#D7263D" if "노인" in acc_type else "#F4A261"
        popup = f"""
        <div style="font-family:Malgun Gothic,Arial;width:360px">
        <h3 style="margin:0 0 6px 0">#{int(row['rank'])} {html.escape(str(row.get('사고지역위치명', '')))}</h3>
        <b>행정동</b>: {html.escape(str(row.get('ADSTRD_NM', '')))}<br>
        <b>사고유형</b>: {html.escape(acc_type)}<br>
        <b>사고건수</b>: {row.get('사고건수', '')}, <b>중상</b>: {row.get('중상자수', '')}, <b>사망</b>: {row.get('사망자수', '')}<br>
        <b>보호구역 거리</b>: {float(row.get('protection_dist_m', 0)):.0f}m<br>
        <b>부족시설</b>: {html.escape(str(row.get('missing_facilities', '')))}<br>
        <b>정책 패키지</b>: {html.escape(str(row.get('policy_package', '')))}<br>
        <b>우선순위</b>: {float(row.get('intervention_priority_index', 0)):.1f}
        </div>
        """
        folium.Marker(
            [point.y, point.x],
            popup=folium.Popup(popup, max_width=430),
            tooltip=f"#{int(row['rank'])} {row.get('ADSTRD_NM', '')}",
            icon=folium.DivIcon(
                html=f"""<div style="background:{color};color:white;border:2px solid #111;border-radius:16px;width:30px;height:30px;text-align:center;line-height:26px;font-weight:800;font-size:12px">{int(row['rank'])}</div>"""
            ),
        ).add_to(groups.get(acc_type, m))

    folium.LayerControl(collapsed=False).add_to(m)
    out = HTML_DIR / "02_top30_policy_map.html"
    m.save(out)
    return out


def fig_to_base64() -> str:
    bio = BytesIO()
    plt.savefig(bio, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    return base64.b64encode(bio.getvalue()).decode("ascii")


def save_static_figures(admin: gpd.GeoDataFrame, accidents: gpd.GeoDataFrame, top30: pd.DataFrame) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(11, 8))
    admin.to_crs(32652).plot(column="max_priority", cmap="YlOrRd", linewidth=0.15, edgecolor="#555", legend=True, ax=ax)
    top_points = accidents[accidents["accident_id"].isin(top30["accident_id"])].to_crs(32652)
    top_points.plot(ax=ax, color="#151515", markersize=16, alpha=0.9)
    ax.set_title("서울 행정동별 보행안전 개입우선순위와 TOP 30", fontsize=18, weight="bold")
    ax.set_axis_off()
    paths["priority_map"] = FIG_DIR / "priority_map.png"
    plt.savefig(paths["priority_map"], dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    top15 = top30.head(15).sort_values("intervention_priority_index")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top15["ADSTRD_NM"].astype(str) + " #" + top15["rank"].astype(str), top15["intervention_priority_index"], color="#C9342B")
    ax.set_xlim(0, 100)
    ax.set_title("TOP 15 개입우선순위", fontsize=17, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    paths["top15_bar"] = FIG_DIR / "top15_priority_bar.png"
    plt.savefig(paths["top15_bar"], dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    ring = top30["protection_ring"].value_counts().reindex(
        ["inside_0_100m", "edge_100_300m", "outside_300_600m", "extended_600_1000m", "beyond_1000m"],
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ring.index, ring.values, color=["#6A994E", "#E7B645", "#E86E2C", "#C9342B", "#7A1E27"])
    ax.set_title("TOP 30의 보호구역 거리 링 분포", fontsize=16, weight="bold")
    ax.tick_params(axis="x", labelrotation=25)
    paths["ring_bar"] = FIG_DIR / "ring_distribution.png"
    plt.savefig(paths["ring_bar"], dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    missing_counts = (
        top30["missing_facilities"]
        .str.split(", ")
        .explode()
        .replace({"상대적으로 충분": np.nan})
        .dropna()
        .value_counts()
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.bar(missing_counts.index, missing_counts.values, color="#264653")
    ax.set_title("TOP 30 부족시설 빈도", fontsize=16, weight="bold")
    ax.tick_params(axis="x", labelrotation=25)
    paths["missing_bar"] = FIG_DIR / "missing_facilities.png"
    plt.savefig(paths["missing_bar"], dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    return paths


def save_dashboard_html(top30: pd.DataFrame, admin: gpd.GeoDataFrame, figure_paths: dict[str, Path]) -> Path:
    def img_b64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    metrics = {
        "사고다발권역": int(admin["accident_area_count"].sum()),
        "분석 행정동": int(len(admin)),
        "A등급 이상 후보": int((top30["priority_grade"].astype(str).str.startswith(("A", "S"))).sum()),
        "TOP30 평균 우선순위": f"{top30['intervention_priority_index'].mean():.1f}",
    }
    top_rows = "\n".join(
        f"<tr><td>{int(r.rank)}</td><td>{html.escape(str(r.ADSTRD_NM))}</td><td>{html.escape(str(r.사고유형구분))}</td><td>{r.intervention_priority_index:.1f}</td><td>{html.escape(str(r.missing_facilities))}</td></tr>"
        for r in top30.head(15).itertuples(index=False)
    )
    cards = "\n".join(f"<div class='card'><b>{k}</b><span>{v}</span></div>" for k, v in metrics.items())
    content = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>보호구역 밖 300m 대시보드</title>
<style>
body{{margin:0;font-family:'Malgun Gothic',Arial,sans-serif;background:#f5f1e8;color:#1c1c1c}}
header{{background:#1d3557;color:white;padding:28px 40px}} h1{{margin:0;font-size:34px}} h2{{margin:24px 0 12px}}
.wrap{{padding:28px 40px}} .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.card{{background:white;border-left:7px solid #e76f51;padding:18px;box-shadow:0 2px 12px #0001}} .card b{{display:block;color:#555}} .card span{{font-size:28px;font-weight:800}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}} img{{width:100%;background:white;border:1px solid #ddd}}
table{{width:100%;border-collapse:collapse;background:white}} th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:left;font-size:13px}} th{{background:#264653;color:white}}
</style></head><body><header><h1>보호구역 밖의 300m</h1><p>서울 어린이·고령자 보행안전 사각지대와 안전시설 우선배치 지도</p></header>
<main class="wrap"><section class="cards">{cards}</section>
<section class="grid"><div><h2>개입우선순위 지도</h2><img src="data:image/png;base64,{img_b64(figure_paths['priority_map'])}"></div>
<div><h2>TOP 15 우선순위</h2><img src="data:image/png;base64,{img_b64(figure_paths['top15_bar'])}"></div>
<div><h2>보호구역 거리 링</h2><img src="data:image/png;base64,{img_b64(figure_paths['ring_bar'])}"></div>
<div><h2>부족시설 빈도</h2><img src="data:image/png;base64,{img_b64(figure_paths['missing_bar'])}"></div></section>
<h2>TOP 15 후보지</h2><table><thead><tr><th>순위</th><th>행정동</th><th>유형</th><th>점수</th><th>부족시설</th></tr></thead><tbody>{top_rows}</tbody></table>
</main></body></html>"""
    out = HTML_DIR / "03_facility_gap_dashboard.html"
    out.write_text(content, encoding="utf-8")
    return out


def save_story_html(top30: pd.DataFrame) -> Path:
    ring_counts = top30["protection_ring"].value_counts().to_dict()
    senior_n = int(top30["사고유형구분"].astype(str).str.contains("노인").sum())
    child_n = int(top30["사고유형구분"].astype(str).str.contains("어린이").sum())
    outside_n = int(top30["protection_ring"].isin(["outside_300_600m", "extended_600_1000m", "beyond_1000m"]).sum())
    content = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>보호구역 밖 300m 스토리</title>
<style>
body{{margin:0;background:#111;color:#f8f4ea;font-family:'Malgun Gothic',Arial,sans-serif}} section{{min-height:85vh;padding:70px 90px;border-bottom:1px solid #333}}
h1{{font-size:56px;line-height:1.1;margin:0 0 20px}} h2{{font-size:38px;color:#ffb703}} p{{font-size:22px;line-height:1.65;max-width:980px}}
.big{{font-size:74px;font-weight:900;color:#e63946}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}} .box{{background:#1f2937;padding:28px;border-radius:18px}}
</style></head><body>
<section><h1>보호구역은 면으로 지정되지만,<br>아이와 노인의 이동은 선으로 이어진다.</h1><p>공식 사고다발지역과 서울시 안전시설 데이터를 결합해 보호구역 밖 생활동선의 공백을 찾았다.</p></section>
<section><h2>핵심 발견</h2><div class="grid"><div class="box"><div class="big">{outside_n}</div><p>TOP 30 중 보호구역 300m 밖 또는 생활확장권 후보</p></div><div class="box"><div class="big">{senior_n}</div><p>고령자 보행사고 중심 후보</p></div><div class="box"><div class="big">{child_n}</div><p>어린이 보행사고 중심 후보</p></div></div></section>
<section><h2>분석 단위</h2><p>내부 0~100m, 경계부 100~300m, 바깥 300m 300~600m, 생활확장권 600~1,000m로 보호공백을 구분했다.</p><p>{html.escape(str(ring_counts))}</p></section>
<section><h2>정책 메시지</h2><p>위험한 곳을 찾는 데서 끝내지 않고, 횡단보도·신호등·과속방지턱·CCTV·보안등·보행자우선도로 중 무엇이 부족한지 후보지별 처방으로 전환했다.</p></section>
</body></html>"""
    out = HTML_DIR / "04_storyboard_summary.html"
    out.write_text(content, encoding="utf-8")
    return out


def main() -> None:
    accidents, admin, top30, top30_gdf = load_data()
    figure_paths = save_static_figures(admin, accidents, top30)
    html_paths = [
        save_priority_map(admin, accidents, top30_gdf),
        save_top30_map(top30_gdf),
        save_dashboard_html(top30, admin, figure_paths),
        save_story_html(top30),
    ]
    print("HTML")
    for path in html_paths:
        print(path)
    print("FIGURES")
    for path in figure_paths.values():
        print(path)


if __name__ == "__main__":
    main()

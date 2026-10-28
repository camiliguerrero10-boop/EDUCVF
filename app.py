"""
SpVAR Colombia – Educación y Crecimiento Económico
Streamlit · Fuentes 100% públicas · Sin API keys
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran
from statsmodels.regression.linear_model import OLS as sm_OLS
from statsmodels.tools.tools import add_constant
from scipy.linalg import inv
from scipy.interpolate import griddata

st.set_page_config(
    page_title="SpVAR Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
h1, h2, h3, p, label, .stMarkdown { color: #e2e8f0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Cargando datos espaciales...")
def load_data():
    url = (
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/"
        "co_2018_MGN_DPTO_POLITICO.geojson"
    )
    gdf = gpd.read_file(url)
    gdf = gdf.rename(columns={"DPTO_CCDGO": "cod_dpto", "DPTO_CNMBR": "departamento"})
    gdf["cod_dpto"] = gdf["cod_dpto"].astype(str).str.zfill(2)
    gdf = gdf.to_crs("EPSG:4326")

    np.random.seed(42)
    N, T = len(gdf), 10
    years = list(range(2015, 2015 + T))

    w_true = Queen.from_dataframe(gdf, use_index=False)
    w_true.transform = "r"

    educ = np.zeros((T, N))
    growth = np.zeros((T, N))
    viirs = np.zeros((T, N))
    educ[0] = 8.2 + 1.8 * np.random.randn(N)
    growth[0] = 2.8 + 1.4 * np.random.randn(N)
    viirs[0] = 12 + 4.5 * np.random.randn(N)

    for t in range(1, T):
        educ[t] = (
            0.62 * educ[t - 1]
            + 0.19 * lag_spatial(w_true, educ[t - 1])
            + 0.09 * growth[t - 1]
            + 0.35 * np.random.randn(N)
        )
        growth[t] = (
            0.51 * growth[t - 1]
            + 0.21 * lag_spatial(w_true, growth[t - 1])
            + 0.28 * educ[t - 1]
            + 0.14 * lag_spatial(w_true, educ[t - 1])
            + 0.85 * np.random.randn(N)
        )
        viirs[t] = (
            0.70 * viirs[t - 1]
            + 0.15 * lag_spatial(w_true, viirs[t - 1])
            + 0.40 * growth[t]
            + 1.1 * np.random.randn(N)
        )

    df_long = pd.DataFrame(
        {
            "year": np.repeat(years, N),
            "departamento": np.tile(gdf["departamento"].values, T),
            "cod_dpto": np.tile(gdf["cod_dpto"].values, T),
            "educacion": educ.flatten(),
            "crecimiento": growth.flatten(),
            "viirs": viirs.flatten(),
        }
    )
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y
    df_long = df_long.merge(gdf[["departamento", "lon", "lat"]], on="departamento", how="left")
    df_long["crecimiento_pos"] = df_long["crecimiento"] - df_long["crecimiento"].min() + 1.5

    # Selección W
    weights = {
        "Queen": Queen.from_dataframe(gdf, use_index=False),
        "Rook": Rook.from_dataframe(gdf, use_index=False),
    }
    for k in [3, 4, 5]:
        weights[f"KNN_{k}"] = KNN.from_dataframe(gdf, k=k, use_index=False)
    for w in weights.values():
        w.transform = "r"

    avg_educ = df_long.groupby("departamento")["educacion"].mean().reindex(gdf["departamento"]).values
    gdf["avg_educ"] = avg_educ

    best_name, best_W, best_abs = "Queen", weights["Queen"], -1
    for name, w in weights.items():
        mi = Moran(gdf["avg_educ"].values, w)
        if mi.p_sim < 0.05 and abs(mi.I) > best_abs:
            best_abs = abs(mi.I)
            best_name, best_W = name, w

    avg_data = (
        df_long.groupby("departamento")
        .agg({"educacion": "mean", "crecimiento": "mean", "viirs": "mean"})
        .reset_index()
    )
    gdf_map = gdf.merge(avg_data, on="departamento", how="left")

    # SpVAR
    df = df_long.sort_values(["departamento", "year"]).copy()
    df["educ_l1"] = df.groupby("departamento")["educacion"].shift(1)
    df["growth_l1"] = df.groupby("departamento")["crecimiento"].shift(1)
    df = df.dropna(subset=["educ_l1", "growth_l1"]).reset_index(drop=True)

    rows = []
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr].set_index("departamento").reindex(gdf["departamento"])
        e1 = sub["educ_l1"].values.astype(float)
        g1 = sub["growth_l1"].values.astype(float)
        vv = sub["viirs"].values.astype(float)
        We = lag_spatial(best_W, e1)
        Wg = lag_spatial(best_W, g1)
        for i, dep in enumerate(gdf["departamento"]):
            rows.append(
                {
                    "year": yr,
                    "departamento": dep,
                    "educacion": float(sub.loc[dep, "educacion"]),
                    "crecimiento": float(sub.loc[dep, "crecimiento"]),
                    "educ_l1": e1[i],
                    "growth_l1": g1[i],
                    "W_educ_l1": We[i],
                    "W_growth_l1": Wg[i],
                    "viirs": vv[i],
                }
            )
    spvar_df = pd.DataFrame(rows)

    def estimate(y, X, names):
        m = sm_OLS(y, add_constant(X)).fit(cov_type="HC1")
        return dict(zip(["const"] + names, m.params))

    coef_e = estimate(
        spvar_df["educacion"].values,
        spvar_df[["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"]].values,
        ["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"],
    )
    coef_g = estimate(
        spvar_df["crecimiento"].values,
        spvar_df[["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"]].values,
        ["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"],
    )

    def impacts(rho, beta, w):
        S = inv(np.eye(w.n) - rho * w.full()[0])
        d = np.mean(np.diag(S)) * beta
        t = np.mean(S.sum(axis=1)) * beta
        return {"direct": d, "indirect": t - d, "total": t}

    imp_e = impacts(coef_e.get("W_educ_l1", 0.0), coef_e["educ_l1"], best_W)
    imp_g = impacts(coef_g.get("W_growth_l1", 0.0), coef_g["growth_l1"], best_W)

    return gdf_map, df_long, spvar_df, best_name, coef_e, coef_g, imp_e, imp_g


gdf_map, df_long, spvar_df, best_W_name, coef_educ, coef_growth, imp_e, imp_g = load_data()

# ---- UI ----
st.title("🇨🇴 SpVAR Colombia")
st.subheader("Educación y Crecimiento Económico · 33 departamentos · 2015-2024")
st.caption("Fuentes 100% públicas · Sin API keys")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Exploración Espacial", "🌌 Visualización 3D", "📊 SpVAR & Mecanismos", "📄 Informe"]
)

with tab1:
    st.markdown("### Mapas coropléticos")
    if st.button("Generar mapas", type="primary", key="m1"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        fig.patch.set_facecolor("#0f172a")
        for ax in axes:
            ax.set_facecolor("#0f172a")
        gdf_map.plot(column="educacion", cmap="YlGnBu", linewidth=0.4, edgecolor="white",
                     legend=True, ax=axes[0], legend_kwds={"shrink": 0.5})
        axes[0].set_title("Educación", fontweight="bold", color="white")
        axes[0].axis("off")
        gdf_map.plot(column="crecimiento", cmap="RdYlGn", linewidth=0.4, edgecolor="white",
                     legend=True, ax=axes[1], legend_kwds={"shrink": 0.5})
        axes[1].set_title("Crecimiento", fontweight="bold", color="white")
        axes[1].axis("off")
        gdf_map.plot(column="viirs", cmap="inferno", linewidth=0.4, edgecolor="white",
                     legend=True, ax=axes[2], legend_kwds={"shrink": 0.5})
        axes[2].set_title("Luces nocturnas", fontweight="bold", color="white")
        axes[2].axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### Mapa interactivo Folium")
    if st.button("Cargar mapa Folium", key="m2"):
        m = folium.Map(location=[4.5, -74.0], zoom_start=5.3, tiles="CartoDB dark_matter")

        def add_layer(col, name, cmap):
            folium.Choropleth(
                geo_data=gdf_map.__geo_interface__,
                data=gdf_map,
                columns=["departamento", col],
                key_on="feature.properties.departamento",
                fill_color=cmap,
                fill_opacity=0.75,
                line_opacity=0.25,
                legend_name=name,
                name=name,
            ).add_to(m)

        add_layer("viirs", "Luces nocturnas", "YlOrRd")
        add_layer("educacion", "Educación", "YlGnBu")
        add_layer("crecimiento", "Crecimiento", "RdYlGn")
        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen().add_to(m)
        st_folium(m, width=900, height=500)

with tab2:
    st.markdown("### Visualizaciones 3D")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Scatter 3D 2024", type="primary", key="b3d"):
            fig = px.scatter_3d(
                df_long[df_long["year"] == 2024],
                x="lon", y="lat", z="viirs",
                color="educacion", size="crecimiento_pos",
                hover_name="departamento",
                color_continuous_scale="Turbo", size_max=22,
                title="Educación × Crecimiento × Luces (2024)",
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", height=520,
                              scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if st.button("Superficie luces", key="bsurf"):
            d = df_long[df_long["year"] == 2024]
            lon, lat, z = d["lon"].values, d["lat"].values, d["viirs"].values
            xi = np.linspace(lon.min(), lon.max(), 70)
            yi = np.linspace(lat.min(), lat.max(), 70)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((lon, lat), z, (Xi, Yi), method="cubic")
            fig = go.Figure(data=[go.Surface(x=Xi, y=Yi, z=Zi, colorscale="Inferno")])
            fig.update_layout(title="Superficie luces nocturnas", template="plotly_dark",
                              paper_bgcolor="#0f172a", height=520,
                              scene=dict(xaxis_title="Lon", yaxis_title="Lat", zaxis_title="VIIRS",
                                         aspectmode="manual", aspectratio=dict(x=1.3, y=1, z=0.45)))
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        if st.button("Animación 2015-2024", key="banim"):
            fig = px.scatter_3d(
                df_long, x="lon", y="lat", z="viirs",
                color="educacion", size="crecimiento_pos",
                animation_frame="year", hover_name="departamento",
                color_continuous_scale="Turbo", size_max=18,
                title="Evolución 2015-2024",
            )
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", height=520,
                              scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)))
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Cuatro mecanismos de transmisión espacial")
    if st.button("Mostrar mecanismos", type="primary", key="bmec"):
        st.markdown(
            f"""
**1. Autodinámica espacial**  
• Educación φ_ee = **{coef_educ['educ_l1']:.4f}**  
• Crecimiento φ_gg = **{coef_growth['growth_l1']:.4f}**

**2. Cruce local**  
• Educ ← Crec local: **{coef_educ['growth_l1']:.4f}**  
• Crec ← Educ local: **{coef_growth['educ_l1']:.4f}**

**3. Relación espacial directa**  
• W·Educ → Educ: **{coef_educ['W_educ_l1']:.4f}**  
• W·Crec → Crec: **{coef_growth['W_growth_l1']:.4f}**

**4. Relación espacial cruzada**  
• W·Crec → Educ: **{coef_educ['W_growth_l1']:.4f}**  
• W·Educ → Crec: **{coef_growth['W_educ_l1']:.4f}**

---
### Impactos de largo plazo (LeSage & Pace)

| Variable | Directo | Indirecto | Total |
|----------|---------|-----------|-------|
| Educación | {imp_e['direct']:.4f} | {imp_e['indirect']:.4f} | {imp_e['total']:.4f} |
| Crecimiento | {imp_g['direct']:.4f} | {imp_g['indirect']:.4f} | {imp_g['total']:.4f} |

**W seleccionada:** `{best_W_name}` · **Obs:** {len(spvar_df)} · **Periodo:** {spvar_df['year'].min()}–{spvar_df['year'].max()}
"""
        )

with tab4:
    st.markdown("### Informe ejecutivo")
    informe = f"""INFORME EJECUTIVO – SpVAR Educación y Crecimiento
Colombia · Departamentos 2015-2024

Fuente geometría: GitHub raw (caticoa3/colombia_mapa)
Matriz W: {best_W_name}
Observaciones: {len(spvar_df)}
Periodo: {spvar_df['year'].min()} – {spvar_df['year'].max()}

MECANISMOS
1. Autodinámica
   Educ φ_ee = {coef_educ['educ_l1']:.4f}
   Crec φ_gg = {coef_growth['growth_l1']:.4f}
2. Cruce local
   Educ ← Crec = {coef_educ['growth_l1']:.4f}
   Crec ← Educ = {coef_growth['educ_l1']:.4f}
3. Espacial directa
   W·Educ → Educ = {coef_educ['W_educ_l1']:.4f}
   W·Crec → Crec = {coef_growth['W_growth_l1']:.4f}
4. Espacial cruzada
   W·Crec → Educ = {coef_educ['W_growth_l1']:.4f}
   W·Educ → Crec = {coef_growth['W_educ_l1']:.4f}

IMPACTOS (LeSage & Pace)
Educación  Directo={imp_e['direct']:.4f} Indirecto={imp_e['indirect']:.4f} Total={imp_e['total']:.4f}
Crecimiento Directo={imp_g['direct']:.4f} Indirecto={imp_g['indirect']:.4f} Total={imp_g['total']:.4f}
"""
    st.download_button(
        "📥 Descargar informe (.txt)",
        data=informe,
        file_name="informe_spvar_colombia.txt",
        mime="text/plain",
        type="primary",
    )
    st.caption("Fuente geometría pública · Datos sintéticos · Sin API keys")

st.markdown(
    '<div style="text-align:center;margin-top:2rem;color:#94a3b8;font-size:0.85rem;">'
    "SpVAR Colombia · Fuentes 100% públicas</div>",
    unsafe_allow_html=True,
)

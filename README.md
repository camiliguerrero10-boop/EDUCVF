# SpVAR Colombia

App Streamlit: Educación y Crecimiento Económico a nivel departamental (modelo SpVAR).

## Streamlit Cloud

1. Sube a un repo de GitHub **solo estos 3 archivos** en la raíz:
   - `app.py`
   - `requirements.txt`
   - `README.md`
2. En [share.streamlit.io](https://share.streamlit.io) → New app
3. Main file: `app.py`
4. Deploy

La primera carga puede tardar 3–5 minutos.

## Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fuentes públicas (sin API keys)

- GeoJSON: https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson
- Tiles: CartoDB dark_matter
- Datos: sintéticos generados en la app

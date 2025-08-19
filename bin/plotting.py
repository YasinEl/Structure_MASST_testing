import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
import pandas as pd
import re
import numpy as np
import json




def raw_data_sankey(df, col1, col2, col3, col4):

    # 1) Collapse to top10 + "others" first (this also converts NaN → "others")
    df = df.copy()
    for col in (col1, col2, col3, col4):
        top10 = df[col].value_counts(dropna=True).nlargest(10).index
        df[col + "_s"] = df[col].where(df[col].isin(top10), "others")

    stages = [col1 + "_s", col2 + "_s", col3 + "_s", col4 + "_s"]

    # 2) Now cast to str (after the collapse)
    df[stages] = df[stages].astype(str)

    # 3) Only now set color_key so its dtype matches the keys you'll build
    df["color_key"] = df[stages[0]]

    # 4) Build labels / idx as you had (these are strings now)
    labels = []
    for i, stg in enumerate(stages, start=1):
        uniques = df[stg].dropna().unique().tolist()
        labels += [f"{i}_{u}" for u in uniques]
    labels = list(dict.fromkeys(labels))

    idx = {}
    for i, stg in enumerate(stages, start=1):
        for u in df[stg].dropna().unique():
            idx[(stg, u)] = labels.index(f"{i}_{u}")

    # 5) Color map: tile palette to exact length (avoids version-dependent length)
    import math, plotly.express as px
    column_1_vals = df[stages[0]].unique().tolist()

    base = px.colors.qualitative.Safe
    palette = (base * math.ceil(len(column_1_vals) / len(base)))[:len(column_1_vals)]
    color_map = dict(zip(column_1_vals, palette))

    # Optional: pin "others" to a neutral gray
    color_map["others"] = "#B0B0B0"

    source, target, value, link_colors = [], [], [], []
    for i in range(len(stages) - 1):
        grp = (
            df
            .dropna(subset=[stages[i], stages[i+1], "color_key"])
            .groupby([stages[i], stages[i+1], "color_key"])
            .size()
            .reset_index(name="count")
        )
        for _, row in grp.iterrows():
            src_val = row[stages[i]]
            tgt_val = row[stages[i+1]]
            color_key = row["color_key"]
            source.append(idx[(stages[i], src_val)])
            target.append(idx[(stages[i+1], tgt_val)])
            value.append(row["count"])
            link_colors.append(color_map.get(color_key, "rgba(0,0,0,0.3)"))

    fig = go.Figure(go.Sankey(
        textfont=dict(family="Arial, sans-serif", size=12, color="black"),
        arrangement="snap",
        node=dict(
            label=labels,
            color=["#F2F2F2"] * len(labels),
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        ),
    ))

    # Add stage annotations
    stage_names = [col1, col2, col3, col4]
    n = len(stage_names) - 1
    for i, name_ in enumerate(stage_names):
        x = i / n
        xanchor = "left" if i == 0 else "right" if i == n else "center"
        fig.add_annotation(
            x=x, y=1.02, xref="paper", yref="paper",
            text=name_, showarrow=False,
            font=dict(size=14, color="black"),
            xanchor=xanchor
        )

    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=60, r=60, t=120, b=20),
    )

    return fig


def export_hits_map(
    df,
    interactive=True,
    out_basename="hit_map",
    max_mri_examples=10,
    env_col="ENVOEnvironmentMaterial",
    # Map engine / style
    engine="mapbox",                 # "mapbox" shows country/city names when zooming
    map_style="open-street-map",     # no token needed; try "carto-positron" too
    projection="natural earth",      # used only when engine="geo"
    # Geo engine decorations
    show_borders=True,
    show_coastlines=True,
    show_graticules=True,
    # Optional admin boundaries overlay (Mapbox engine only)
    admin_geojson=None,              # dict or path to GeoJSON with borders
    admin_line_color="rgba(80,80,80,0.7)",
    admin_line_width=1,
    admin_opacity=0.8,
    # Hover content
    hover_mri="count",               # "none" | "count" | "examples"
):
    """
    Build a world map of hit locations from a DataFrame with columns:
      - 'mri'
      - 'LatitudeandLongitude' formatted as 'lat|lon', e.g. '32.876878|-117.234459'
      - env_col (default: 'ENVOEnvironmentMaterial') used for color

    Exports:
      - PNG:  <out_basename>.png   (requires 'kaleido')
      - HTML: <out_basename>.html  (if interactive=True; mouse wheel zoom enabled)

    hover_mri:
      - "none"     → no MRI info in hover
      - "count"    → only show hit count
      - "examples" → show up to max_mri_examples examples
    """

    required = {"LatitudeandLongitude", "mri", env_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")

    # Filter env & parse coords
    df = df[df[env_col].notna() & (df[env_col] != "missing value")].copy()

    latlon = df["LatitudeandLongitude"].astype(str).str.extract(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([+-]?\d+(?:\.\d+)?)\s*$"
    )
    lat = pd.to_numeric(latlon[0], errors="coerce")
    lon = pd.to_numeric(latlon[1], errors="coerce")
    valid = lat.between(-90, 90) & lon.between(-180, 180)

    df2 = df.loc[valid].copy()
    if df2.empty:
        fig = (px.scatter_geo(projection=projection) if engine=="geo"
               else px.scatter_mapbox(lat=[], lon=[], mapbox_style=map_style, zoom=1))
        fig.update_layout(
            title="Hit Locations (no valid rows)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        if engine == "geo":
            fig.update_geos(
                showland=True, landcolor="#f6f6f6",
                showocean=True, oceancolor="#eef6ff",
                showcountries=show_borders, countrycolor="rgba(80,80,80,0.5)",
                showcoastlines=show_coastlines, coastlinecolor="rgba(80,80,80,0.5)",
                lataxis=dict(showgrid=show_graticules, gridcolor="rgba(0,0,0,0.15)", gridwidth=0.5),
                lonaxis=dict(showgrid=show_graticules, gridcolor="rgba(0,0,0,0.15)", gridwidth=0.5),
            )
        return fig, df2

    df2["lat"] = lat.loc[valid].values
    df2["lon"] = lon.loc[valid].values

    # Aggregate to one point per (lat, lon, env_col)
    def _join_examples(s):
        if hover_mri != "examples":
            return ""  # unused
        u = pd.Series(s.astype(str).unique())
        return ", ".join(u.iloc[:max_mri_examples]) + (f", … (+{len(u)-max_mri_examples} more)" if len(u) > max_mri_examples else "")

    hits = (
        df2.groupby(["lat", "lon", env_col], as_index=False)
           .agg(n=("mri", "count"), mri_examples=("mri", _join_examples))
           .sort_values("n", ascending=False)
           .reset_index(drop=True)
    )

    cat_order = hits.groupby(env_col)["n"].sum().sort_values(ascending=False).index.tolist()
    center = {"lat": float(hits["lat"].mean()), "lon": float(hits["lon"].mean())}

    # Build figure with tightly controlled hover
    hover_data = {env_col: False, "lat": False, "lon": False, "n": False, "mri_examples": False}
    if engine == "geo":
        fig = px.scatter_geo(
            hits, lat="lat", lon="lon",
            size="n", size_max=18,
            color=env_col, category_orders={env_col: cat_order},
            projection=projection,
            hover_name=None, hover_data=hover_data,
        )
    else:
        fig = px.scatter_mapbox(
            hits, lat="lat", lon="lon",
            size="n", size_max=20,
            color=env_col, category_orders={env_col: cat_order},
            mapbox_style=map_style, center=center, zoom=1,
            hover_name=None, hover_data=hover_data,
        )

    # Compose hovertemplate according to hover_mri
    if hover_mri == "none":
        cd = hits[[env_col]].values
        htmpl = "<b>%{customdata[0]}</b><br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
    elif hover_mri == "count":
        cd = hits[[env_col, "n"]].values
        htmpl = "<b>%{customdata[0]}</b><br>Hits: %{customdata[1]}<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<extra></extra>"
    else:  # "examples"
        cd = hits[[env_col, "n", "mri_examples"]].values
        htmpl = "<b>%{customdata[0]}</b><br>Hits: %{customdata[1]}<br>Lat: %{lat:.5f}<br>Lon: %{lon:.5f}<br>MRIs: %{customdata[2]}<extra></extra>"

    fig.update_traces(marker=dict(opacity=0.9), customdata=cd, hovertemplate=htmpl)
    fig.update_layout(
        title=f"Hit Locations by {env_col} (total hits={int(hits['n'].sum())}, points={len(hits)})",
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="closest",
        legend_title_text=env_col,
    )

    # Geo engine decorations
    if engine == "geo":
        fig.update_geos(
            showland=True, landcolor="#f6f6f6",
            showocean=True, oceancolor="#eef6ff",
            showcountries=show_borders, countrycolor="rgba(80,80,80,0.5)",
            showcoastlines=show_coastlines, coastlinecolor="rgba(80,80,80,0.5)",
            lataxis=dict(showgrid=show_graticules, gridcolor="rgba(0,0,0,0.15)", gridwidth=0.5),
            lonaxis=dict(showgrid=show_graticules, gridcolor="rgba(0,0,0,0.15)", gridwidth=0.5),
        )
    else:
        # Optional borders overlay for mapbox
        if admin_geojson is not None:
            if isinstance(admin_geojson, str):
                with open(admin_geojson, "r", encoding="utf-8") as f:
                    admin_geojson = json.load(f)
            fig.update_layout(
                mapbox_layers=[
                    dict(
                        sourcetype="geojson",
                        source=admin_geojson,
                        type="line",
                        color=admin_line_color,
                        line=dict(width=admin_line_width),
                        opacity=admin_opacity,
                        below="traces",
                    )
                ]
            )


    return fig, hits

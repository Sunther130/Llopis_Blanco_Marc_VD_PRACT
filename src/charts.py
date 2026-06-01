"""Mòdul per a la creació de gràfics i visualitzacions a partir del dataset de Steam."""

import plotly.express as px
import numpy as np
import pandas as pd

from settings import (
    TEMPLATE,
    TITLE_FONT,
    CHART_FONT,
    SCATTER_FILLCOLOR,
    HRECT_SCATTER_OPACITY,
    SCATTER_OPACITY_CIRCLE,
    TAG_SCALE_LOGARITMICA,
    TAG_SCALE_LINEAL,
)


LIGHT_TREEMAP_SCALE = [[0, '#f3f8fc'], [0.35, '#dce8f5'], [0.7, '#aed6f1'], [1, '#66a9d0']]
TOP_GENRE_COLORS = [
    "#00a2ff",  # Action
    "#ff0000",  # Indie
    "#00cd55",  # Adventure
    "#ffb300",  # RPG
    "#c96ab1",  # Casual
    "#6dd400",  # Simulation
    "#0019fc",  # Strategy
    "#8b4513",  # Massively
    "#b300ff",  # Sports
    "#000000",  # Racing
]


def _format_genre_name(genre_col: str) -> str:
    return genre_col.replace("Genre_", "").replace("_", " ")


def _get_genre_sales_series(steam_df: pd.DataFrame, genre_cols: list) -> pd.Series:
    return steam_df[genre_cols].multiply(steam_df["Owners_mid"], axis=0).sum()


def _get_top_selling_genres(
        steam_df: pd.DataFrame,
        genre_cols: list,
        n: int = 10) -> tuple[list, list]:
    top_genre_cols = _get_genre_sales_series(steam_df, genre_cols).nlargest(n).index.tolist()
    top_genre_names = [_format_genre_name(col) for col in top_genre_cols]
    return top_genre_cols, top_genre_names


def _get_top_genre_palette(top_genre_names: list) -> dict:
    return dict(zip(top_genre_names, TOP_GENRE_COLORS[:len(top_genre_names)]))


def _apply_top_genre_palette(fig, palette: dict):
    for trace in fig.data:
        if trace.name in palette:
            trace.update(marker=dict(color=palette[trace.name]), legendgroup=trace.name)
    return fig


def _filter_top_selling_genres(
        df: pd.DataFrame,
        top_genre_cols: list,
        label_col: str) -> pd.DataFrame:
    filtered = df.iloc[0:0].copy() if df.empty or not top_genre_cols else df.copy()
    filtered[label_col] = pd.Series(dtype="object")

    if filtered.empty:
        return filtered

    matrix = filtered[top_genre_cols].to_numpy(dtype=float)
    idx = np.argmax(matrix, axis=1)
    has_top_genre = matrix.max(axis=1) > 0.5
    top_genre_names = np.array([_format_genre_name(col) for col in top_genre_cols])

    filtered = filtered.loc[has_top_genre].copy()
    filtered[label_col] = top_genre_names[idx[has_top_genre]]
    return filtered


def _apply_custom_theme(fig):
    fig.update_layout(
        font=CHART_FONT,
        title_font=TITLE_FONT,
        title_x=0.02,
        title_xanchor="left",
    )
    return fig


def _apply_light_treemap_theme(fig):
    fig.update_layout(
        template="plotly_white",
        height=480,
        coloraxis_showscale=False,
        margin=dict(t=72, r=16, b=16, l=16),
    )
    fig.update_traces(
        textinfo="label+value",
        textfont=CHART_FONT,
        marker=dict(cornerradius=5),
        selector=dict(type="treemap"),
    )
    return _apply_custom_theme(fig)


def build_genre_count_treemap(
        steam_df: pd.DataFrame,
        genre_cols: list,
        n: int = 10):
    counts = steam_df[genre_cols].sum().reset_index()
    counts.columns = ["Gènere", "Total jocs"]
    counts["Gènere"] = (
        counts["Gènere"]
        .str.replace("Genre_", "", regex=False)
        .str.replace("_", " ")
    )
    top = counts.nlargest(n, "Total jocs")
    max_val = top["Total jocs"].max()
    top["Color_Group"] = ["Més gran" if val == max_val else "Altres" for val in top["Total jocs"]]

    fig = px.treemap(
        top,
        path=["Gènere"],
        values="Total jocs",
        color="Color_Group",
        color_discrete_map={"Més gran": "#1f77b4", "Altres": "#d3d3d3"},
        title=f"Els {n} gèneres més presents a Steam",
    )

    return _apply_light_treemap_theme(fig)


def build_genre_sales_treemap(
        steam_df: pd.DataFrame,
        genre_cols: list,
        n: int = 10):
    sales = _get_genre_sales_series(steam_df, genre_cols).rename_axis("Genre_Column")
    sales = sales.reset_index(name="Vendes estimades")
    sales["Gènere"] = sales["Genre_Column"].map(_format_genre_name)
    top = sales.nlargest(n, "Vendes estimades")[["Gènere", "Vendes estimades"]].copy()
    max_val = top["Vendes estimades"].max()
    top["Color_Group"] = ["Més gran" if val == max_val else "Altres" for val in top["Vendes estimades"]]

    fig = px.treemap(
        top,
        path=["Gènere"],
        values="Vendes estimades",
        color="Color_Group",
        color_discrete_map={"Més gran": "#0d28d8", "Altres": "#d3d3d3"},
        title=f"Els {n} gèneres més venuts a Steam",
    )
    return _apply_light_treemap_theme(fig)


def build_price_scatter(
        steam_df: pd.DataFrame,
        genre_cols: list,
        threshold: int = 2000,
        sample_size: int = 200):
    df = steam_df[
        (steam_df["Total_reviews"] >= threshold)
        & steam_df["Price"].notna()
        & (steam_df["Price"] > 0)
    ].copy()

    top_genre_cols, top_genre_names = _get_top_selling_genres(steam_df, genre_cols)
    palette = _get_top_genre_palette(top_genre_names)
    df = _filter_top_selling_genres(df, top_genre_cols, "Genre_focus")

    df = df[df["Satisfaction_ratio"].between(0, 1)].copy()
    dfs = [
        frame.sample(n=min(len(frame), sample_size), random_state=42)
        for _, frame in df.groupby("Genre_focus")
    ]
    if dfs:
        df = pd.concat(dfs).reset_index(drop=True)
    df["Bubble_size"] = np.sqrt(df["Total_reviews"].clip(lower=1))

    fig = px.scatter(
        df,
        x="Price",
        y="Satisfaction_ratio",
        color="Genre_focus",
        size_max=16,
        log_x=False,
        opacity=SCATTER_OPACITY_CIRCLE,
        render_mode="webgl",
        hover_name="Name",
        hover_data={
            "Price": ":.2f",
            "Satisfaction_ratio": ":.0%",
            "Total_reviews": ":,",
            "Peak CCU": ":,",
            "Bubble_size": False,
        },
        labels={
            "Price": "Preu $",
            "Satisfaction_ratio": "Rati de satisfacció",
            "Genre_focus": "Gènere",
        },
        category_orders={"Genre_focus": top_genre_names},
        color_discrete_map=palette,
        title=(
            "Satisfacció dels usuaris segons el preu"
        ),
    )
    fig.update_traces(marker=dict(line=dict(width=0)))

    fig.update_layout(
        template=TEMPLATE,
        height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(t=100, r=30, b=60, l=40),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(range=[0, 1], gridcolor="rgba(0,0,0,0.06)", tickformat=".0%"),
    )
    fig.add_hrect(y0=0.80, y1=1.0, fillcolor=SCATTER_FILLCOLOR, opacity=HRECT_SCATTER_OPACITY)

    return _apply_custom_theme(_apply_top_genre_palette(fig, palette))


def build_ccu_scatter(
        steam_df: pd.DataFrame,
        genre_cols: list,
        log_x: bool = True):
    df = steam_df[
        (steam_df["Peak CCU"] > 100) & (steam_df["Total_reviews"] >= 100)
    ].copy()
    top_genre_cols, top_genre_names = _get_top_selling_genres(steam_df, genre_cols)
    palette = _get_top_genre_palette(top_genre_names)
    df = _filter_top_selling_genres(df, top_genre_cols, "Genre_focus")

    x_label = (
        "Nombre d'usuaris concurrents (escala log)"
        if log_x
        else "Nombre d'usuaris concurrents (escala lineal)"
    )
    fig = px.scatter(
        df,
        x="Peak CCU",
        y="Satisfaction_ratio",
        size="Recommendations",
        size_max=50,
        opacity=SCATTER_OPACITY_CIRCLE,
        color="Genre_focus",
        hover_name="Name",
        hover_data={
            "Peak CCU": ":,.0f",
            "Satisfaction_ratio": ":.1%",
            "Recommendations": ":,.0f",
            "Total_reviews": ":,.0f",
        },
        log_x=log_x,
        title="Recepció del joc per part dels jugadors",
        labels={
            "Peak CCU": x_label,
            "Satisfaction_ratio": "Rati de satisfacció",
            "Genre_focus": "Gènere",
        },
        category_orders={"Genre_focus": top_genre_names},
        color_discrete_map=palette,
    )
    fig.add_hrect(
        y0=0.80,
        y1=1.0,
        fillcolor=SCATTER_FILLCOLOR,
        opacity=HRECT_SCATTER_OPACITY,
    )
    fig.update_layout(
        template=TEMPLATE,
        height=600,
        margin=dict(t=100, r=30, b=110, l=40),
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        legend=dict(font=CHART_FONT),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                active=0 if log_x else 1,
                buttons=[
                    dict(
                        label=TAG_SCALE_LOGARITMICA,
                        method="relayout",
                        args=[
                            {
                                "xaxis.type": "log",
                                "xaxis.title.text": "Nombre d'usuaris concurrents (escala log)",
                                "xaxis.autorange": True,
                            }
                        ],
                    ),
                    dict(
                        label=TAG_SCALE_LINEAL,
                        method="relayout",
                        args=[
                            {
                                "xaxis.type": "linear",
                                "xaxis.title.text": "Nombre d'usuaris concurrents (escala lineal)",
                                "xaxis.autorange": True,
                            }
                        ],
                    ),
                ],
                showactive=True,
                x=1,
                xanchor="right",
                y=-0.2,
                yanchor="middle",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
                font=CHART_FONT,
                pad=dict(r=4, t=0, b=0, l=4),
            )
        ],
    )
    return _apply_custom_theme(_apply_top_genre_palette(fig, palette))


def build_profile_scatter(
        steam_df: pd.DataFrame,
        log_x: bool = True,
        ccu_min: int = 1000,
        reviews_min: int = 1000):
    df = steam_df[
        (steam_df["Total_reviews"] >= reviews_min) & (steam_df["Peak CCU"] > ccu_min)
    ].copy()

    high_pop_threshold = df["Peak CCU"].quantile(0.85)

    def classify(row):
        high_pop = row["Peak CCU"] > high_pop_threshold
        high_sat = row["Satisfaction_ratio"] >= 0.90
        low_sat = row["Satisfaction_ratio"] < 0.50
        if high_sat and not high_pop:
            return "Hidden Gem"
        if high_sat and high_pop:
            return "Blockbuster"
        if low_sat and high_pop:
            return "Polaritzat"
        if low_sat and not high_pop:
            return "Decebedor"
        return "Mainstream"

    df["Profile"] = df.apply(classify, axis=1)
    x_label = (
        "Nombre d'usuaris concurrents (escala log)"
        if log_x
        else "Nombre d'usuaris concurrents (escala lineal)"
    )
    fig = px.scatter(
        df,
        x="Peak CCU",
        y="Satisfaction_ratio",
        color="Profile",
        hover_name="Name",
        hover_data={
            "Peak CCU": ":,.0f",
            "Satisfaction_ratio": ":.1%",
            "Total_reviews": ":,.0f",
            "Genre_principal": True,
        },
        log_x=log_x,
        opacity=0.6,
        title="Mapa de perfils segons recepció",
        labels={
            "Peak CCU": x_label,
            "Satisfaction_ratio": "Rati de satisfacció",
        },
        color_discrete_map={
            "Hidden Gem": "#d4ac0d",
            "Blockbuster": "#1a88c9",
            "Polaritzat": "#27ae60",
            "Mainstream": "#7f8c8d",
            "Decebedor": "#e74c3c",
        },
    )
    fig.update_layout(
        template=TEMPLATE,
        height=550,
        margin=dict(t=100, r=30, b=110, l=40),
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        legend=dict(font=CHART_FONT, title_text="Perfil"),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                active=0 if log_x else 1,
                buttons=[
                    dict(
                        label=TAG_SCALE_LOGARITMICA,
                        method="relayout",
                        args=[
                            {
                                "xaxis.type": "log",
                                "xaxis.title.text": "Nombre d'usuaris concurrents (escala log)",
                                "xaxis.autorange": True,
                            }
                        ],
                    ),
                    dict(
                        label=TAG_SCALE_LINEAL,
                        method="relayout",
                        args=[
                            {
                                "xaxis.type": "linear",
                                "xaxis.title.text": "Nombre d'usuaris concurrents (escala lineal)",
                                "xaxis.autorange": True,
                            }
                        ],
                    ),
                ],
                showactive=True,
                x=1,
                xanchor="right",
                y=-0.2,
                yanchor="middle",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
                font=CHART_FONT,
                pad=dict(r=4, t=0, b=0, l=4),
            )
        ],
    )
    return _apply_custom_theme(fig), df


def build_retention_scatter(
        steam_df: pd.DataFrame,
        min_playtime: int = 60,
        log_x: bool = True):
    df = steam_df[
        (steam_df["Average playtime forever"] > min_playtime)
        & (steam_df["Median playtime forever"] > 0)
        & (steam_df["Average playtime two weeks"].notna())
    ].copy()

    x_label = (
        "Intensitat d'ús total (min, escala log)"
        if log_x
        else "Intensitat d'ús total (min, escala lineal)"
    )
    fig = px.scatter(
        df,
        x="Average playtime forever",
        y="Retention_ratio",
        color="Skewness",
        color_continuous_scale="Blues",
        opacity=0.55,
        log_x=log_x,
        hover_name="Name",
        hover_data={
            "Average playtime forever": ":,.0f",
            "Retention_ratio": ":.1%",
            "Skewness": ":.1f",
            "Median playtime forever": ":,.0f",
        },
        labels={
            "Average playtime forever": x_label,
            "Retention_ratio": "Rati de retenció recent",
            "Skewness": "",
        },
        title="Patrons de retenció i intensitat d'ús",
    )

    fig.add_hline(y=0.5, line_dash="dot", line_color="grey", opacity=0.5)
    fig.add_vline(
        x=df["Average playtime forever"].median(),
        line_dash="dot", line_color="grey", opacity=0.5,
    )

    fig.update_layout(
        template=TEMPLATE,
        height=600,
        margin=dict(t=100, r=30, b=110, l=40),
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                active=0 if log_x else 1,
                buttons=[
                    dict(
                        label=TAG_SCALE_LOGARITMICA,
                        method="relayout",
                        args=[
                            {
                                "xaxis.type": "log",
                                "xaxis.title.text": "Intensitat d'ús total (min, escala log)",
                                "xaxis.autorange": True,
                            }
                        ],
                    ),
                    dict(
                        label=TAG_SCALE_LINEAL,
                        method="relayout",
                        args=[
                            {
                                "xaxis.type": "linear",
                                "xaxis.title.text": "Intensitat d'ús total (min, escala lineal)",
                                "xaxis.autorange": True,
                            }
                        ],
                    ),
                ],
                showactive=True,
                x=1,
                xanchor="right",
                y=-0.2,
                yanchor="middle",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
                font=CHART_FONT,
                pad=dict(r=4, t=0, b=0, l=4),
            )
        ],
    )
    return _apply_custom_theme(fig)

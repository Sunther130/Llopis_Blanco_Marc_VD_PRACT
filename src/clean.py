"""Mòdul per a la neteja i preprocessament de les dades del dataset de Steam."""


import re
import pandas as pd
import numpy as np

from settings import DATASET_PATH, DATASET_PARQUET_PATH


def clean_release_dates(steam_df: pd.DataFrame) -> pd.DataFrame:
    # Conversió de la data de llançament
    steam_df['Release date'] = pd.to_datetime(steam_df['Release date'], format='mixed', errors='coerce')
    steam_df['Release_year'] = steam_df['Release date'].dt.year

    # Filtrar anys raonables (es descarten registres anteriors al 2014 per baixa fiabilitat de dades)
    steam_df = steam_df[(steam_df['Release_year'] >= 2014)]

    return steam_df


def convert_numeric_columns(steam_df: pd.DataFrame) -> pd.DataFrame:
    # Tractar valors numèrics
    numeric_cols = ['Price', 'Peak CCU', 'Positive', 'Negative', 'User score', 
                    'Metacritic score', 'Recommendations', 'Average playtime forever',
                    'Median playtime forever', 'Average playtime two weeks', 
                    'Median playtime two weeks']

    for col in numeric_cols:
        steam_df[col] = pd.to_numeric(steam_df[col], errors='coerce')

    return steam_df


def encode_genres(steam_df: pd.DataFrame) -> pd.DataFrame:
    steam_df['Genres'] = steam_df['Genres'].apply(lambda x: re.sub(r',?Early Access', '', str(x)) if pd.notna(x) else x)
    steam_df['Genres'] = steam_df['Genres'].apply(lambda x: re.sub(r',?Free To Play', '', str(x)) if pd.notna(x) else x)

    # One-hot encoding multi-etiqueta: cada gènere de la llista rep la seva pròpia columna binària
    genre_dummies = (
        steam_df['Genres']
        .fillna('')
        .str.split(',')
        .explode()
        .str.strip()
        .loc[lambda s: s != '']
        .pipe(lambda s: pd.get_dummies(s, prefix='Genre'))
        .groupby(level=0)
        .max()
        .astype(int)
    )
    genre_dummies.columns = [col.replace(' ', '_') for col in genre_dummies.columns]
    steam_df = pd.concat([steam_df, genre_dummies], axis=1)

    return steam_df


def estimate_owners_midpoint(steam_df: pd.DataFrame) -> pd.DataFrame:
    def parse_owners(x):
        try:
            parts = str(x).replace(',', '').split(' - ')
            return (int(parts[0]) + int(parts[1])) / 2
        except (ValueError, IndexError, AttributeError):
            return np.nan

    steam_df['Owners_mid'] = steam_df['Estimated owners'].apply(parse_owners)

    return steam_df


def create_derived_metrics(steam_df: pd.DataFrame) -> pd.DataFrame:
    # Percentatge de valoracions positives sobre el total
    steam_df['Total_reviews'] = steam_df['Positive'] + steam_df['Negative']
    steam_df['Satisfaction_ratio'] = steam_df['Positive'] / steam_df['Total_reviews']
    steam_df['Satisfaction_ratio'] = steam_df['Satisfaction_ratio'].replace([np.inf, -np.inf], np.nan)
    steam_df["Retention_ratio"] = (steam_df["Average playtime two weeks"] / steam_df["Average playtime forever"]).clip(0, 1)
    steam_df["Skewness"] = (steam_df["Average playtime forever"] / steam_df["Median playtime forever"]).clip(1, 20)

    return steam_df


def preprocess_data() -> None:
    # Càrrega del dataset
    steam_df = pd.read_csv(DATASET_PATH, low_memory=False)
    steam_df = clean_release_dates(steam_df)
    steam_df = convert_numeric_columns(steam_df)
    steam_df = encode_genres(steam_df)
    steam_df = estimate_owners_midpoint(steam_df)
    steam_df = create_derived_metrics(steam_df)

    steam_df.to_parquet(DATASET_PARQUET_PATH, index=False)


if __name__ == '__main__':
    preprocess_data()

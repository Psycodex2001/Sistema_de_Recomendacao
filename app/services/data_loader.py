"""
Serviço de carregamento de dados.

Estratégias em ordem de preferência:
1. Usar a biblioteca `datasets` (HuggingFace) com data_files explícito.
2. Baixar os CSVs via HTTP direto do HuggingFace Hub.
3. Levantar RuntimeError com mensagem informativa.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Tuple

import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# URL base dos arquivos raw no HuggingFace
_HF_RAW_BASE = (
    "https://huggingface.co/datasets/ashraq/ml-latest-small/resolve/main"
)

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


# ──────────────────────────────────────────────────────────────────────────────
# Utilitários
# ──────────────────────────────────────────────────────────────────────────────

def _parse_year(title: str) -> Tuple[str, int | None]:
    """Extrai o ano entre parênteses do título do filme."""
    m = _YEAR_RE.search(title)
    if m:
        year = int(m.group(1))
        clean = title[: m.start()].strip()
        return clean, year
    return title, None


def _standardize_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas para snake_case e valida tipos."""
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df[["user_id", "movie_id", "rating", "timestamp"]]


def _standardize_movies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"movieId": "movie_id"})
    df["movie_id"] = df["movie_id"].astype(int)
    return df[["movie_id", "title", "genres"]]


def _standardize_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df["movie_id"] = df["movie_id"].astype(int)
    df = df[["movie_id", "tag", "timestamp"]].dropna(subset=["tag"])
    df["tag"] = df["tag"].astype(str)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Download / leitura
# ──────────────────────────────────────────────────────────────────────────────

def _load_via_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estratégia 1: biblioteca datasets do HuggingFace."""
    from datasets import load_dataset  # type: ignore

    logger.info("Estratégia 1: carregando via datasets HuggingFace…")
    ds = load_dataset(
        "ashraq/ml-latest-small",
        data_files={
            "ratings": "ratings.csv",
            "movies": "movies.csv",
            "tags": "tags.csv",
        },
    )
    ratings_df = _standardize_ratings(ds["ratings"].to_pandas())
    movies_df = _standardize_movies(ds["movies"].to_pandas())
    tags_df = _standardize_tags(ds["tags"].to_pandas())
    return ratings_df, movies_df, tags_df


def _load_via_http() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estratégia 2: download HTTP direto dos CSVs."""
    import requests  # type: ignore

    logger.info("Estratégia 2: baixando CSVs via HTTP…")

    def fetch(filename: str) -> pd.DataFrame:
        url = f"{_HF_RAW_BASE}/{filename}"
        logger.info(f"  GET {url}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

    ratings_df = _standardize_ratings(fetch("ratings.csv"))
    movies_df = _standardize_movies(fetch("movies.csv"))
    try:
        tags_df = _standardize_tags(fetch("tags.csv"))
    except Exception as exc:
        logger.warning(f"Não foi possível baixar tags.csv ({exc}). Usando DataFrame vazio.")
        tags_df = pd.DataFrame(columns=["movie_id", "tag", "timestamp"])
    return ratings_df, movies_df, tags_df


def _download_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tenta as estratégias em ordem e retorna os DataFrames."""
    try:
        return _load_via_datasets()
    except Exception as e:
        logger.warning(f"Estratégia 1 falhou: {e}")

    try:
        return _load_via_http()
    except Exception as e:
        logger.error(f"Estratégia 2 falhou: {e}")

    raise RuntimeError(
        "Não foi possível baixar o dataset MovieLens. "
        "Verifique sua conexão com a internet e tente novamente."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inserção no banco de dados
# ──────────────────────────────────────────────────────────────────────────────

def load_and_populate_db(db: Session) -> None:
    """
    Verifica se o banco já foi populado; caso contrário, baixa o dataset e
    insere todos os dados.

    Args:
        db: Sessão SQLAlchemy ativa.
    """
    from app.models.db_models import Movie, Rating, Tag, User

    # Verifica se os dados já existem
    if db.query(Movie).count() > 0:
        logger.info("Banco já populado, pulando download do dataset.")
        return

    logger.info("Populando banco de dados com o dataset MovieLens Small…")
    ratings_df, movies_df, tags_df = _download_raw_data()

    # ── Insere Filmes ──────────────────────────────────────────────────────
    logger.info(f"Inserindo {len(movies_df)} filmes…")
    movies_to_insert = []
    for _, row in movies_df.iterrows():
        clean_title, year = _parse_year(str(row["title"]))
        movies_to_insert.append(
            Movie(
                movie_id=int(row["movie_id"]),
                title=clean_title,
                genres=str(row["genres"]),
                year=year,
            )
        )
    db.bulk_save_objects(movies_to_insert)
    db.commit()
    logger.info("Filmes inseridos.")

    # ── Insere Usuários (únicos do dataset) ───────────────────────────────
    unique_user_ids = sorted(ratings_df["user_id"].unique())
    logger.info(f"Inserindo {len(unique_user_ids)} usuários…")
    users_to_insert = [
        User(username=f"ml_user_{uid}")
        for uid in unique_user_ids
    ]
    db.bulk_save_objects(users_to_insert)
    db.commit()

    # Monta mapeamento user_id (ML) → id (DB)
    db_users = {int(u.username.split("_")[-1]): u.id for u in db.query(User).all()}
    db_movies = {m.movie_id: m.id for m in db.query(Movie).all()}

    # ── Insere Avaliações ──────────────────────────────────────────────────
    logger.info(f"Inserindo {len(ratings_df)} avaliações…")
    ratings_to_insert = []
    seen = set()
    for _, row in ratings_df.iterrows():
        uid = int(row["user_id"])
        mid = int(row["movie_id"])
        key = (uid, mid)
        if key in seen:
            continue
        seen.add(key)
        db_uid = db_users.get(uid)
        db_mid = db_movies.get(mid)
        if db_uid and db_mid:
            ratings_to_insert.append(
                Rating(
                    user_id=db_uid,
                    movie_id=db_mid,
                    rating=float(row["rating"]),
                    timestamp=int(row.get("timestamp", 0) or 0),
                )
            )
    db.bulk_save_objects(ratings_to_insert)
    db.commit()
    logger.info("Avaliações inseridas.")

    # ── Insere Tags ────────────────────────────────────────────────────────
    if not tags_df.empty:
        logger.info(f"Inserindo {len(tags_df)} tags…")
        tags_to_insert = []
        for _, row in tags_df.iterrows():
            mid = int(row["movie_id"])
            db_mid = db_movies.get(mid)
            if db_mid:
                tags_to_insert.append(
                    Tag(
                        movie_id=db_mid,
                        tag=str(row["tag"])[:200],
                        timestamp=int(row.get("timestamp", 0) or 0),
                    )
                )
        db.bulk_save_objects(tags_to_insert)
        db.commit()
        logger.info("Tags inseridas.")

    logger.info("✅ Banco de dados populado com sucesso!")

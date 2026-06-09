"""
Schemas Pydantic para validação de request/response da API.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# Usuários
# ──────────────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=100, examples=["joao_silva"])


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ──────────────────────────────────────────────────────────────────────────────
# Filmes
# ──────────────────────────────────────────────────────────────────────────────

class MovieCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300, examples=["Inception (2010)"])
    genres: str = Field(
        default="(no genres listed)",
        examples=["Action|Sci-Fi|Thriller"],
        description="Gêneros separados por pipe '|'",
    )


class MovieResponse(BaseModel):
    id: int
    movie_id: int
    title: str
    genres: str
    year: Optional[int] = None

    @property
    def genres_list(self) -> List[str]:
        if not self.genres or self.genres == "(no genres listed)":
            return []
        return self.genres.split("|")

    model_config = {"from_attributes": True}


class MovieSearchResponse(BaseModel):
    total: int
    movies: List[MovieResponse]


# ──────────────────────────────────────────────────────────────────────────────
# Avaliações
# ──────────────────────────────────────────────────────────────────────────────

class RatingCreate(BaseModel):
    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0, description="movie_id do filme (ID MovieLens)")
    rating: float = Field(..., ge=0.5, le=5.0, description="Nota de 0.5 a 5.0")

    @field_validator("rating")
    @classmethod
    def validate_half_step(cls, v: float) -> float:
        """Garante incrementos de 0,5 como no MovieLens."""
        rounded = round(v * 2) / 2
        if abs(rounded - v) > 1e-9:
            raise ValueError("rating deve ser múltiplo de 0.5 (ex.: 0.5, 1.0, 1.5, …, 5.0)")
        return rounded


class RatingResponse(BaseModel):
    id: int
    user_id: int
    movie_id: int
    rating: float

    model_config = {"from_attributes": True}


class PreferenceUpdate(BaseModel):
    """Atualiza ou cria múltiplas avaliações de uma vez (preferências do usuário)."""
    ratings: List[RatingCreate] = Field(
        ...,
        min_length=1,
        examples=[[
            {"user_id": 1, "movie_id": 1, "rating": 4.5},
            {"user_id": 1, "movie_id": 3, "rating": 3.0},
        ]],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Recomendações
# ──────────────────────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: str
    year: Optional[int] = None
    score: float = Field(..., description="Pontuação híbrida normalizada [0,1]")
    source: str = Field(
        ...,
        description="Origem: 'hybrid', 'collaborative', 'content_based' ou 'popular'",
    )


class RecommendationResponse(BaseModel):
    user_id: int
    n_recommendations: int
    model_trained: bool
    recommendations: List[RecommendationItem]


# ──────────────────────────────────────────────────────────────────────────────
# Status do modelo
# ──────────────────────────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    is_trained: bool
    n_users: int
    n_movies: int
    n_ratings: int
    cf_weight: float
    cb_weight: float
    svd_components: int
    message: str

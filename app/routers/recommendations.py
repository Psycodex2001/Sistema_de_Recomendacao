"""
Router de Recomendações.

Endpoints:
    GET  /recommendations/{user_id}         → Recomendações personalizadas
    POST /recommendations/model/retrain     → Força re-treino do modelo
    GET  /recommendations/model/status      → Informações e status do modelo
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.db_models import Movie, Rating, User
from app.models.schemas import ModelStatus, RecommendationItem, RecommendationResponse
from app.state import app_state

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_user_or_404(user_id: int, db: Session) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"Usuário {user_id} não encontrado.")
    return user


def _get_rated_movies(user_id: int, db: Session) -> dict:
    """Retorna {movie_id (ML): rating} para um usuário."""
    rows = (
        db.query(Rating, Movie)
        .join(Movie, Rating.movie_id == Movie.id)
        .filter(Rating.user_id == user_id)
        .all()
    )
    return {movie.movie_id: rating.rating for rating, movie in rows}


# ──────────────────────────────────────────────────────────────────────────────
# Recomendações personalizadas
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{user_id}",
    response_model=RecommendationResponse,
    summary="Recomendações personalizadas para um usuário",
)
def get_recommendations(
    user_id: int,
    n: int = Query(
        default=settings.DEFAULT_TOP_N,
        ge=1,
        le=settings.MAX_TOP_N,
        description="Número de recomendações",
    ),
    db: Session = Depends(get_db),
):
    """
    Retorna as `n` melhores recomendações de filmes para o usuário informado.

    **Estratégia híbrida:**
    - Usuário com avaliações: combina Filtragem Colaborativa (SVD) +
      Filtragem por Conteúdo (TF-IDF) com pesos configuráveis.
    - Usuário sem avaliações (cold-start): retorna os filmes mais populares
      calculados pela média bayesiana.

    O campo `source` em cada item indica a origem da recomendação:
    `hybrid`, `collaborative`, `content_based` ou `popular`.
    """
    _get_user_or_404(user_id, db)

    model_trained = app_state.recommender is not None and app_state.recommender.is_trained

    if not model_trained:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não treinado. Aguarde a inicialização do sistema.",
        )

    # Histórico de avaliações do usuário
    rated_movies = _get_rated_movies(user_id, db)

    # Gera recomendações
    raw = app_state.recommender.recommend(
        user_id=user_id,
        rated_movies=rated_movies,
        n=n,
    )

    # Enriquece com metadados dos filmes
    items = []
    for rec in raw:
        movie = db.query(Movie).filter(Movie.movie_id == rec["movie_id"]).first()
        if not movie:
            continue
        items.append(
            RecommendationItem(
                movie_id=movie.movie_id,
                title=movie.title,
                genres=movie.genres,
                year=movie.year,
                score=round(rec["score"], 6),
                source=rec["source"],
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        n_recommendations=len(items),
        model_trained=model_trained,
        recommendations=items,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gestão do modelo
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/model/retrain",
    summary="Força o re-treinamento do modelo",
    response_model=ModelStatus,
)
async def retrain_model(db: Session = Depends(get_db)):
    """
    Re-treina o modelo híbrido com todos os dados presentes no banco.

    Útil após adicionar muitas avaliações novas. O processo é executado
    em background para não bloquear o event loop.
    """
    if app_state.recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Instância do recomendador não inicializada.",
        )

    logger.info("Re-treino manual solicitado…")

    # Executa em thread para não bloquear o event loop do FastAPI
    await asyncio.to_thread(app_state.recommender.train_from_db, db)

    return _build_model_status(db)


@router.get(
    "/model/status",
    response_model=ModelStatus,
    summary="Status e métricas do modelo",
)
def get_model_status(db: Session = Depends(get_db)):
    """Retorna informações sobre o estado atual do modelo de recomendação."""
    return _build_model_status(db)


def _build_model_status(db: Session) -> ModelStatus:
    """Monta o objeto ModelStatus a partir do estado atual."""
    from app.models.db_models import Rating, Movie, User

    n_users = db.query(User).count()
    n_movies = db.query(Movie).count()
    n_ratings = db.query(Rating).count()

    rec = app_state.recommender
    is_trained = rec is not None and rec.is_trained

    return ModelStatus(
        is_trained=is_trained,
        n_users=n_users,
        n_movies=n_movies,
        n_ratings=n_ratings,
        cf_weight=rec.cf_weight if rec else settings.CF_WEIGHT,
        cb_weight=rec.cb_weight if rec else settings.CB_WEIGHT,
        svd_components=settings.N_SVD_COMPONENTS,
        message=(
            "Modelo pronto para uso."
            if is_trained
            else "Modelo ainda não treinado. Aguarde a inicialização."
        ),
    )

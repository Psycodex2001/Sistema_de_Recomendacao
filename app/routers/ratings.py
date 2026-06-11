"""
Router de Avaliações.

Endpoints:
    POST /ratings/          → Adiciona ou atualiza avaliação (user + filme)
    GET  /ratings/          → Lista avaliações com filtros opcionais
    DELETE /ratings/{id}    → Remove avaliação por ID interno
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Movie, Rating, User
from app.models.schemas import RatingCreate, RatingResponse

router = APIRouter()


@router.post(
    "/",
    response_model=RatingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Adiciona ou atualiza uma avaliação",
)
def add_or_update_rating(payload: RatingCreate, db: Session = Depends(get_db)):
    """
    Registra a avaliação de um usuário para um filme.

    - Se a combinação (user_id, movie_id) já existir, a nota é **atualizada**.
    - `movie_id` deve ser o **ID do MovieLens** (campo `movie_id` dos filmes).
    - `rating` deve ser um múltiplo de 0,5 entre 0,5 e 5,0.
    """
    # Valida usuário
    user = db.query(User).filter(User.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"Usuário {payload.user_id} não encontrado.")

    # Localiza filme pelo movie_id MovieLens
    movie = db.query(Movie).filter(Movie.movie_id == payload.movie_id).first()
    if not movie:
        raise HTTPException(
            status_code=404,
            detail=f"Filme com movie_id={payload.movie_id} não encontrado.",
        )

    # Upsert
    existing = (
        db.query(Rating)
        .filter(Rating.user_id == payload.user_id, Rating.movie_id == movie.id)
        .first()
    )
    if existing:
        existing.rating = payload.rating
        db.commit()
        db.refresh(existing)
        rating = existing
    else:
        rating = Rating(
            user_id=payload.user_id,
            movie_id=movie.id,
            rating=payload.rating,
        )
        db.add(rating)
        db.commit()
        db.refresh(rating)

    return RatingResponse(
        id=rating.id,
        user_id=rating.user_id,
        movie_id=movie.movie_id,
        rating=rating.rating,
    )


@router.get(
    "/",
    response_model=List[RatingResponse],
    summary="Lista avaliações",
)
def list_ratings(
    user_id: Optional[int] = Query(None, description="Filtrar por usuário"),
    movie_id: Optional[int] = Query(None, description="Filtrar por movie_id (MovieLens)"),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Lista avaliações com filtros opcionais por `user_id` e/ou `movie_id`.
    """
    query = db.query(Rating)

    if user_id is not None:
        query = query.filter(Rating.user_id == user_id)

    if movie_id is not None:
        movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
        if not movie:
            return []
        query = query.filter(Rating.movie_id == movie.id)

    ratings = query.limit(limit).all()

    result = []
    for r in ratings:
        movie = db.query(Movie).filter(Movie.id == r.movie_id).first()
        result.append(
            RatingResponse(
                id=r.id,
                user_id=r.user_id,
                movie_id=movie.movie_id if movie else r.movie_id,
                rating=r.rating,
            )
        )
    return result


@router.delete(
    "/{rating_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove uma avaliação",
)
def delete_rating(rating_id: int, db: Session = Depends(get_db)):
    """Remove uma avaliação pelo seu ID interno."""
    rating = db.query(Rating).filter(Rating.id == rating_id).first()
    if not rating:
        raise HTTPException(status_code=404, detail=f"Avaliação {rating_id} não encontrada.")
    db.delete(rating)
    db.commit()

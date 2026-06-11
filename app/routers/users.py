"""
Router de Usuários — CRUD completo.

Endpoints:
    POST   /users/                      → Cria usuário
    GET    /users/{user_id}             → Retorna usuário
    GET    /users/{user_id}/ratings     → Lista avaliações do usuário
    PUT    /users/{user_id}/preferences → Atualiza preferências (batch de avaliações)
    DELETE /users/{user_id}             → Remove usuário
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Movie, Rating, User
from app.models.schemas import (
    PreferenceUpdate,
    RatingResponse,
    UserCreate,
    UserResponse,
)

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_user_or_404(user_id: int, db: Session) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"Usuário {user_id} não encontrado.")
    return user


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Cria novo usuário",
)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Cria um novo usuário no sistema. O username deve ser único."""
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Username '{payload.username}' já está em uso.",
        )
    user = User(username=payload.username)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Retorna dados de um usuário",
)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Retorna as informações de um usuário pelo seu ID."""
    return _get_user_or_404(user_id, db)


@router.get(
    "/{user_id}/ratings",
    response_model=List[RatingResponse],
    summary="Lista avaliações de um usuário",
)
def get_user_ratings(user_id: int, db: Session = Depends(get_db)):
    """Retorna todas as avaliações feitas por um usuário específico."""
    _get_user_or_404(user_id, db)
    ratings = db.query(Rating).filter(Rating.user_id == user_id).all()
    # Converte foreign key movie_id (DB) para movie_id (MovieLens)
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


@router.put(
    "/{user_id}/preferences",
    response_model=List[RatingResponse],
    summary="Atualiza preferências do usuário (batch de avaliações)",
)
def update_preferences(
    user_id: int,
    payload: PreferenceUpdate,
    db: Session = Depends(get_db),
):
    """
    Cria ou atualiza múltiplas avaliações de um usuário de uma só vez.
    Se a avaliação (user + movie) já existir, sobrescreve a nota.
    """
    _get_user_or_404(user_id, db)

    updated: List[RatingResponse] = []

    for rating_in in payload.ratings:
        if rating_in.user_id != user_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"user_id no body ({rating_in.user_id}) difere do "
                    f"user_id na URL ({user_id})."
                ),
            )

        # Localiza o filme pelo movie_id MovieLens
        movie = db.query(Movie).filter(Movie.movie_id == rating_in.movie_id).first()
        if not movie:
            raise HTTPException(
                status_code=404,
                detail=f"Filme com movie_id={rating_in.movie_id} não encontrado.",
            )

        # Upsert
        existing_rating = (
            db.query(Rating)
            .filter(Rating.user_id == user_id, Rating.movie_id == movie.id)
            .first()
        )
        if existing_rating:
            existing_rating.rating = rating_in.rating
            db.commit()
            db.refresh(existing_rating)
            r = existing_rating
        else:
            r = Rating(user_id=user_id, movie_id=movie.id, rating=rating_in.rating)
            db.add(r)
            db.commit()
            db.refresh(r)

        updated.append(
            RatingResponse(
                id=r.id,
                user_id=r.user_id,
                movie_id=movie.movie_id,
                rating=r.rating,
            )
        )

    return updated


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove um usuário",
)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Remove um usuário e todas as suas avaliações."""
    user = _get_user_or_404(user_id, db)
    db.delete(user)
    db.commit()

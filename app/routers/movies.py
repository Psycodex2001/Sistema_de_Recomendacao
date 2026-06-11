"""
Router de Filmes.

Endpoints:
    GET  /movies/           → Lista filmes (paginado, filtro por gênero)
    GET  /movies/search     → Busca por título
    GET  /movies/{movie_id} → Detalhes de um filme
    POST /movies/           → Adiciona novo filme
    GET  /movies/{movie_id}/similar → Filmes similares (CB)
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.db_models import Movie
from app.models.schemas import MovieCreate, MovieResponse, MovieSearchResponse
from app.state import app_state

router = APIRouter()


def _movie_to_schema(movie: Movie) -> MovieResponse:
    return MovieResponse(
        id=movie.id,
        movie_id=movie.movie_id,
        title=movie.title,
        genres=movie.genres,
        year=movie.year,
    )


@router.get(
    "/",
    response_model=MovieSearchResponse,
    summary="Lista filmes",
)
def list_movies(
    skip: int = Query(0, ge=0, description="Registros a pular"),
    limit: int = Query(20, ge=1, le=100, description="Máximo de registros"),
    genre: Optional[str] = Query(None, description="Filtrar por gênero (ex.: Action)"),
    db: Session = Depends(get_db),
):
    """Lista filmes com paginação e filtro opcional por gênero."""
    query = db.query(Movie)
    if genre:
        query = query.filter(Movie.genres.like(f"%{genre}%"))
    total = query.count()
    movies = query.offset(skip).limit(limit).all()
    return MovieSearchResponse(total=total, movies=[_movie_to_schema(m) for m in movies])


@router.get(
    "/search",
    response_model=MovieSearchResponse,
    summary="Busca filmes por título",
)
def search_movies(
    q: str = Query(..., min_length=1, description="Termos de busca"),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Busca filmes cujo título contenha a string informada (case-insensitive)."""
    terms = q.strip().split()
    query = db.query(Movie)
    for term in terms:
        query = query.filter(Movie.title.ilike(f"%{term}%"))
    movies = query.limit(limit).all()
    return MovieSearchResponse(total=len(movies), movies=[_movie_to_schema(m) for m in movies])


@router.get(
    "/{movie_id}",
    response_model=MovieResponse,
    summary="Detalhes de um filme",
)
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    """
    Retorna os detalhes de um filme.

    O `movie_id` deve corresponder ao **ID do MovieLens**, não ao ID interno do banco.
    """
    movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail=f"Filme {movie_id} não encontrado.")
    return _movie_to_schema(movie)


@router.post(
    "/",
    response_model=MovieResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Adiciona novo filme",
)
def add_movie(payload: MovieCreate, db: Session = Depends(get_db)):
    """
    Adiciona um filme ao catálogo.

    O `movie_id` é gerado automaticamente (max existente + 1).
    """
    # Gera próximo movie_id
    max_movie = db.query(Movie).order_by(Movie.movie_id.desc()).first()
    next_id = (max_movie.movie_id + 1) if max_movie else 1

    # Extrai ano do título se presente
    from app.services.data_loader import _parse_year
    clean_title, year = _parse_year(payload.title)

    movie = Movie(
        movie_id=next_id,
        title=clean_title,
        genres=payload.genres,
        year=year,
    )
    db.add(movie)
    db.commit()
    db.refresh(movie)
    return _movie_to_schema(movie)


@router.get(
    "/{movie_id}/similar",
    response_model=List[MovieResponse],
    summary="Filmes similares ao informado (filtragem por conteúdo)",
)
def get_similar_movies(
    movie_id: int,
    n: int = Query(10, ge=1, le=50, description="Número de similares a retornar"),
    db: Session = Depends(get_db),
):
    """
    Retorna os `n` filmes mais similares ao filme indicado,
    calculado com base em gêneros e tags (TF-IDF + cosseno).
    """
    # Confirma que o filme existe
    movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail=f"Filme {movie_id} não encontrado.")

    if app_state.recommender is None or not app_state.recommender.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não treinado. Aguarde a inicialização.",
        )

    similares = app_state.recommender.cb.get_similar_movies(movie_id, top_n=n)
    if not similares:
        return []

    result: List[MovieResponse] = []
    for sim_movie_id, _score in sorted(similares.items(), key=lambda x: x[1], reverse=True):
        m = db.query(Movie).filter(Movie.movie_id == sim_movie_id).first()
        if m:
            result.append(_movie_to_schema(m))
    return result

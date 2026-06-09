"""
Modelos ORM (SQLAlchemy) que mapeiam as tabelas do banco de dados.
"""

from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    """Usuário do sistema de recomendação."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.id} username={self.username!r}>"


class Movie(Base):
    """Filme catalogado no sistema."""
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    # movie_id é o ID original do MovieLens (usado pelo modelo)
    movie_id = Column(Integer, unique=True, index=True, nullable=False)
    title = Column(String(300), index=True, nullable=False)
    genres = Column(String(500), nullable=False, default="(no genres listed)")
    year = Column(Integer, nullable=True)

    ratings = relationship("Rating", back_populates="movie", cascade="all, delete-orphan")
    tags = relationship("Tag", back_populates="movie", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Movie movie_id={self.movie_id} title={self.title!r}>"


class Rating(Base):
    """Avaliação de um usuário para um filme (0,5 – 5,0 em incrementos de 0,5)."""
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    rating = Column(Float, nullable=False)
    timestamp = Column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie"),
    )

    user = relationship("User", back_populates="ratings")
    movie = relationship("Movie", back_populates="ratings")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Rating user={self.user_id} movie={self.movie_id} rating={self.rating}>"


class Tag(Base):
    """Tag textual atribuída por um usuário a um filme."""
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    tag = Column(String(200), nullable=False)
    timestamp = Column(Integer, nullable=True)

    movie = relationship("Movie", back_populates="tags")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tag movie={self.movie_id} tag={self.tag!r}>"

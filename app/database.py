"""
Configuração do banco de dados com SQLAlchemy.
Usa SQLite por padrão; troque DATABASE_URL no .env para PostgreSQL em produção.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()

# Garante que o diretório de dados exista
os.makedirs("data", exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # Necessário para SQLite
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency FastAPI: fornece sessão de banco de dados e fecha ao final."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

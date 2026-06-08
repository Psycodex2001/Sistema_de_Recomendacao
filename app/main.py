"""
Ponto de entrada da aplicação FastAPI — CineMatch.

Fluxo de inicialização (lifespan):
1. Cria as tabelas do banco de dados.
2. Popula o banco com o dataset MovieLens Small (se ainda não populado).
3. Treina o modelo híbrido de recomendação.
4. Disponibiliza a API.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import Base, SessionLocal, engine
from app.recommender.hybrid import HybridRecommender
from app.routers import movies, ratings, recommendations, users
from app.services.data_loader import load_and_populate_db
from app.state import app_state

# ── Configuração de logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia inicialização e encerramento da aplicação."""
    logger.info("=" * 60)
    logger.info(f"  Iniciando {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    # 1. Cria tabelas no banco
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("Tabelas do banco verificadas/criadas.")

    # 2. Popula banco com MovieLens (se necessário) — I/O bound, roda em thread
    db = SessionLocal()
    try:
        await asyncio.to_thread(load_and_populate_db, db)
    finally:
        db.close()

    # 3. Inicializa e treina o modelo híbrido — CPU bound, roda em thread
    app_state.recommender = HybridRecommender(
        cf_weight=settings.CF_WEIGHT,
        cb_weight=settings.CB_WEIGHT,
        n_components=settings.N_SVD_COMPONENTS,
    )

    db = SessionLocal()
    try:
        await asyncio.to_thread(app_state.recommender.train_from_db, db)
    finally:
        db.close()

    app_state.is_ready = True
    logger.info("=" * 60)
    logger.info("  ✅  CineMatch pronto! Acesse /docs para a documentação.")
    logger.info("=" * 60)

    yield  # ← Aplicação fica em execução aqui

    logger.info("Encerrando CineMatch…")


# ── Aplicação FastAPI ─────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (permite requisições do frontend ou ferramentas como Postman)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(users.router,           prefix="/users",           tags=["Usuários"])
app.include_router(movies.router,          prefix="/movies",          tags=["Filmes"])
app.include_router(ratings.router,         prefix="/ratings",         tags=["Avaliações"])
app.include_router(recommendations.router, prefix="/recommendations", tags=["Recomendações"])


# ── Endpoints raiz ────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Verificação de saúde")
def root():
    """Retorna status básico da API."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"], summary="Status detalhado")
def health():
    """Retorna status detalhado incluindo se o modelo já foi treinado."""
    return {
        "status": "healthy" if app_state.is_ready else "initializing",
        "model_trained": (
            app_state.recommender is not None and app_state.recommender.is_trained
        ),
    }

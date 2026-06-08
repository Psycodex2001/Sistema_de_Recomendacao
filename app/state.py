"""
Estado global da aplicação.
Mantém o modelo de recomendação treinado acessível por todos os routers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.recommender.hybrid import HybridRecommender


@dataclass
class AppState:
    recommender: "HybridRecommender | None" = field(default=None)
    is_ready: bool = False


# Instância única (singleton) compartilhada pela aplicação
app_state = AppState()

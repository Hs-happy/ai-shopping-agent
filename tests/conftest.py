from __future__ import annotations

import pytest

from app.agent.runner import ShoppingAgent
from app.config import get_settings
from app.retrieval.engine import SearchEngine
from app.store.db import Database


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def engine(settings):
    db = Database(settings.db_path)
    if db.count_products() == 0:
        pytest.skip("商品库为空，先执行 python -m app.data.generator 与 python -m app.data.etl")
    return SearchEngine(settings, db=db).ensure_ready()


@pytest.fixture(scope="session")
def agent(settings, engine):
    return ShoppingAgent(settings, engine=engine)

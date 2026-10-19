from __future__ import annotations

import pandas as pd
import pytest

from forecast_optimize.data.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="session")
def small_data() -> pd.DataFrame:
    """About 16 months of hourly data: big enough to train, small enough to be fast."""
    return generate(SyntheticConfig(start="2023-01-01", end="2024-04-30 23:00", seed=11))


@pytest.fixture(scope="session")
def fast_lgbm() -> dict[str, object]:
    return {"n_estimators": 60, "learning_rate": 0.15, "num_leaves": 15}

"""Shared fixtures for the InterveneAI test suite (Phase 12)."""

from pathlib import Path

import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 42
SAMPLE_N = 500


def _pq(*parts):
    return pd.read_parquet(ROOT.joinpath(*parts))


@pytest.fixture(scope="session")
def orders():
    df = _pq("data", "processed", "customer_orders.parquet")
    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )
    return df


@pytest.fixture(scope="session")
def snapshots():
    df = _pq("data", "processed", "customer_snapshots.parquet")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


@pytest.fixture(scope="session")
def features_v2():
    df = _pq("data", "processed", "customer_features_v2.parquet")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


@pytest.fixture(scope="session")
def features_v3():
    df = _pq("data", "processed", "customer_features_v3.parquet")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


@pytest.fixture(scope="session")
def experiment():
    df = _pq(
        "data", "synthetic", "intervention_experiment.parquet"
    )
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def sample_rows(df, n=SAMPLE_N, seed=SEED):
    return df.sample(
        min(n, len(df)), random_state=seed
    ).reset_index(drop=True)

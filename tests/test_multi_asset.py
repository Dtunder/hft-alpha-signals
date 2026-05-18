import pytest
import torch
import os
import json
from src.multi_asset_brain import MultiAssetBrain
from src.backtest_engine import BacktestEngine

def test_multi_asset_brain_initialization():
    brain = MultiAssetBrain(num_assets=5, window_size=10)
    assert brain.num_assets == 5
    assert brain.window_size == 10

    # Check if JIT model is saved
    assert os.path.exists("brain/multi_asset_core_traced.pt")

def test_multi_asset_brain_update_warmup():
    brain = MultiAssetBrain(num_assets=3, window_size=5)

    # Initial updates should return None until window is full
    for _ in range(4):
        corr, breakouts, latency = brain.update([1.0, 2.0, 3.0])
        assert corr is None
        assert breakouts is None
        assert latency == 0.0

def test_multi_asset_brain_inference():
    brain = MultiAssetBrain(num_assets=3, window_size=5)

    for i in range(5):
        corr, breakouts, latency = brain.update([i*1.0, i*2.0, i*3.0])

    # At 5th tick, we should get predictions
    assert corr is not None
    assert breakouts is not None
    assert latency > 0.0

    # Correlation matrix should be 3x3
    assert corr.shape == (3, 3)
    # Breakout scores should be length 3
    assert breakouts.shape == (3,)

def test_backtest_engine_execution():
    # Keep it very small for quick test
    engine = BacktestEngine(num_assets=3, ticks=15, window_size=5)
    engine.run()

    # Check if report is generated
    assert os.path.exists("logs/backtest_report.json")

    with open("logs/backtest_report.json", "r") as f:
        report = json.load(f)

    assert "metrics" in report
    assert "total_trades" in report["metrics"]
    assert "average_latency_us" in report["metrics"]

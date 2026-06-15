import pytest
import random
from src.alpha_brain import HFTAlphaSignals

def test_alpha_signals_contract():
    brain = HFTAlphaSignals(obi_threshold=0.70)
    bids = [[100, 10], [99, 10]]
    asks = [[101, 10], [102, 10]]
    signal, obi = brain.check_signals(bids, asks)
    assert signal in ["BUY", "SELL", "HOLD"]
    assert isinstance(obi, float)

def test_funding_rate_bias():
    brain = HFTAlphaSignals(obi_threshold=0.70)
    # Prime the history
    for _ in range(3):
        brain.check_signals([[100, 100]], [[101, 1]]) # OBI > 0
    # Now this would be a BUY, but funding_rate > 0.01%
    signal, obi = brain.check_signals([[100, 100]], [[101, 1]], funding_rate=0.00015)
    assert signal == "HOLD"

def test_momentum_filter():
    brain = HFTAlphaSignals(obi_threshold=0.70)
    # 1. First tick strong BUY, but history empty -> HOLD
    signal, _ = brain.check_signals([[100, 100]], [[101, 1]])
    assert signal == "HOLD"
    # 2. Second tick strong BUY -> HOLD
    signal, _ = brain.check_signals([[100, 100]], [[101, 1]])
    assert signal == "HOLD"
    # 3. Third tick strong BUY -> HOLD
    signal, _ = brain.check_signals([[100, 100]], [[101, 1]])
    assert signal == "HOLD"
    # 4. Fourth tick strong BUY -> BUY
    signal, _ = brain.check_signals([[100, 100]], [[101, 1]])
    assert signal == "BUY"

def test_win_rate_simulation():
    brain = HFTAlphaSignals(obi_threshold=0.70)
    wins = 0
    total_signals = 0

    current_skew = 0.0
    last_signal = "HOLD"

    random.seed(42)
    for i in range(10000):
        # Update skew with some momentum
        current_skew = 0.85 * current_skew + 0.15 * random.uniform(-1, 1)

        bid_vol = max(1.0, 10.0 + current_skew * 100)
        ask_vol = max(1.0, 10.0 - current_skew * 100)

        bids = [[100, bid_vol]]
        asks = [[101, ask_vol]]

        signal, obi = brain.check_signals(bids, asks)

        if last_signal == "BUY":
            if obi > 0: wins += 1
            total_signals += 1
        elif last_signal == "SELL":
            if obi < 0: wins += 1
            total_signals += 1

        last_signal = signal

    win_rate = wins / total_signals if total_signals > 0 else 0
    print(f"Simulation win rate: {win_rate} ({wins}/{total_signals})")
    assert win_rate > 0.60
# Final test explicit trigger

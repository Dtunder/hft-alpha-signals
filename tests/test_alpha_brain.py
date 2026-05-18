import pytest
from src.alpha_brain import HFTAlphaSignals

def test_obi_calculation():
    brain = HFTAlphaSignals()
    bids = [[100, 10], [99, 5]]
    asks = [[101, 5], [102, 5]]
    # total bids volume = 15
    # total asks volume = 10
    # OBI = (15 - 10) / 25 = 5 / 25 = 0.2
    assert pytest.approx(brain.analyze_order_book(bids, asks)) == 0.2

def test_threshold_bounds():
    brain_low = HFTAlphaSignals(obi_threshold=0.5)
    assert brain_low.obi_threshold == 0.6

    brain_high = HFTAlphaSignals(obi_threshold=0.9)
    assert brain_high.obi_threshold == 0.75

    brain_mid = HFTAlphaSignals(obi_threshold=0.65)
    assert brain_mid.obi_threshold == 0.65

def test_momentum_filter_upward():
    brain = HFTAlphaSignals(obi_threshold=0.75, momentum_window=3)

    # Tick 1: Mid = 100.5
    bids = [[100, 10]]
    asks = [[101, 10]]
    signal, obi = brain.check_signals(bids, asks)
    assert signal == "HOLD"

    # Tick 2: Mid = 101.5
    bids = [[101, 10]]
    asks = [[102, 10]]
    signal, obi = brain.check_signals(bids, asks)
    assert signal == "HOLD"

    # Tick 3: Mid = 102.5, Strong OBI = 1.0
    bids = [[102, 100]]
    asks = [[103, 0]]
    signal, obi = brain.check_signals(bids, asks)

    # Momentum is upward (100.5 -> 101.5 -> 102.5) and OBI > 0.75 -> BUY
    assert signal == "BUY"

def test_momentum_filter_fake_spike():
    brain = HFTAlphaSignals(obi_threshold=0.75, momentum_window=3)

    # Tick 1: Mid = 100.5
    bids = [[100, 10]]
    asks = [[101, 10]]
    brain.check_signals(bids, asks)

    # Tick 2: Mid = 99.5
    bids = [[99, 10]]
    asks = [[100, 10]]
    brain.check_signals(bids, asks)

    # Tick 3: Mid = 98.5, Strong OBI = 1.0 (but downward momentum)
    bids = [[98, 100]]
    asks = [[99, 0]]
    signal, obi = brain.check_signals(bids, asks)

    # Momentum is downward (100.5 -> 99.5 -> 98.5) but OBI is highly positive.
    # This should be a fake OBI spike.
    assert signal == "HOLD (Fake OBI Spike)"

def test_momentum_filter_downward():
    brain = HFTAlphaSignals(obi_threshold=0.75, momentum_window=3)

    # Tick 1: Mid = 100.5
    bids = [[100, 10]]
    asks = [[101, 10]]
    brain.check_signals(bids, asks)

    # Tick 2: Mid = 99.5
    bids = [[99, 10]]
    asks = [[100, 10]]
    brain.check_signals(bids, asks)

    # Tick 3: Mid = 98.5, Strong negative OBI = -1.0
    bids = [[98, 0]]
    asks = [[99, 100]]
    signal, obi = brain.check_signals(bids, asks)

    # Momentum is downward and OBI < -0.75 -> SELL
    assert signal == "SELL"

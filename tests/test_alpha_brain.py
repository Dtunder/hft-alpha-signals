import random
import pytest
from src.alpha_brain import HFTAlphaSignals

def test_interface_contract():
    brain = HFTAlphaSignals(obi_threshold=0.70)
    bids = [[58000, 100], [57999, 100], [57998, 100]]
    asks = [[58001, 10], [58002, 10], [58003, 10]]

    # Send 3 times to bypass momentum hold
    brain.check_signals(bids, asks)
    brain.check_signals(bids, asks)
    signal, obi = brain.check_signals(bids, asks)

    assert signal in ["BUY", "SELL", "HOLD"]
    assert isinstance(obi, float)
    assert obi > 0.70
    assert signal == "BUY"

def test_funding_rate_suppression():
    brain = HFTAlphaSignals(obi_threshold=0.70)
    bids = [[58000, 100], [57999, 100], [57998, 100]]
    asks = [[58001, 10], [58002, 10], [58003, 10]]

    # Try with high funding rate (> 0.01% i.e. 0.0001)
    brain.check_signals(bids, asks, funding_rate=0.0002)
    brain.check_signals(bids, asks, funding_rate=0.0002)
    signal, obi = brain.check_signals(bids, asks, funding_rate=0.0002)

    assert signal == "HOLD"  # Should be suppressed

    # Try with low funding rate (<= 0.0001)
    # The queue already has positive OBI from previous calls, so it will immediately trigger
    signal, obi = brain.check_signals(bids, asks, funding_rate=0.00005)
    assert signal == "BUY"  # Should not be suppressed

def test_momentum_filter():
    brain = HFTAlphaSignals(obi_threshold=0.70)

    # Huge bid wall, but only one tick (history < 3)
    bids = [[58000, 1000]]
    asks = [[58001, 10]]

    signal, obi = brain.check_signals(bids, asks)
    assert signal == "HOLD"

    # Second tick, huge bid wall
    signal, obi = brain.check_signals(bids, asks)
    assert signal == "HOLD"

    # Third tick, huge bid wall
    signal, obi = brain.check_signals(bids, asks)
    assert signal == "BUY"

    # Switch to huge ask wall. Must wait 3 ticks to change to SELL
    bids2 = [[58000, 10]]
    asks2 = [[58001, 1000]]

    signal, obi = brain.check_signals(bids2, asks2)
    assert signal == "HOLD"  # Mix of positive and negative in history

    signal, obi = brain.check_signals(bids2, asks2)
    assert signal == "HOLD"

    signal, obi = brain.check_signals(bids2, asks2)
    assert signal == "SELL"  # All 3 history items are negative now

def generate_random_order_book(bias="NONE"):
    bid_vol_base = 50
    ask_vol_base = 50

    if bias == "UP":
        bid_vol_base = 200
        ask_vol_base = 10
    elif bias == "DOWN":
        bid_vol_base = 10
        ask_vol_base = 200

    bids = [[58000 - i, random.uniform(1.0, 10.0) * bid_vol_base] for i in range(5)]
    asks = [[58001 + i, random.uniform(1.0, 10.0) * ask_vol_base] for i in range(5)]

    return bids, asks

def test_win_rate_simulation():
    brain = HFTAlphaSignals(obi_threshold=0.70)

    trades = 0
    wins = 0

    # Simulate a correlated market where OBI actually predicts the next price movement
    # We will feed the brain a sequence of ticks.
    # To hit a "win", the price movement after a signal must match the signal.
    # We'll artificially create momentum (UP/DOWN/NONE) spanning several ticks.

    # We'll run 10000 ticks.

    current_momentum = "NONE"
    ticks_in_momentum = 0

    for i in range(10000):
        # Change momentum randomly
        if ticks_in_momentum <= 0:
            current_momentum = random.choice(["UP", "DOWN", "NONE", "NONE", "NONE"])
            ticks_in_momentum = random.randint(3, 10)

        bids, asks = generate_random_order_book(current_momentum)

        signal, obi = brain.check_signals(bids, asks, funding_rate=0.0)

        if signal in ["BUY", "SELL"]:
            trades += 1
            # If our signal correctly identified the underlying momentum, it's a win
            if signal == "BUY" and current_momentum == "UP":
                wins += 1
            elif signal == "SELL" and current_momentum == "DOWN":
                wins += 1
            # If momentum was NONE, it's a 50/50 toss up (just to be conservative)
            elif current_momentum == "NONE":
                if random.random() > 0.5:
                    wins += 1

        ticks_in_momentum -= 1

    # We need to ensure we took some trades
    assert trades > 0

    win_rate = wins / trades
    print(f"Total Trades: {trades}, Wins: {wins}, Win Rate: {win_rate:.2f}")

    # Target win rate > 0.60
    assert win_rate > 0.60

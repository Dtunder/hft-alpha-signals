import pytest
import time
from src.cross_arb import CrossExchangeArbitrageDetector

def test_bybit_lag_arbitrage():
    detector = CrossExchangeArbitrageDetector(threshold=0.0)

    # Base timestamp
    current_time = int(time.time() * 1000)

    # Simulate a sudden price jump on Binance and OKX
    # Binance and OKX update with high prices: Bid 105, Ask 106
    detector.update_price("Binance", bid=105.0, ask=106.0, timestamp=current_time)
    detector.update_price("OKX", bid=105.0, ask=106.0, timestamp=current_time)

    # Bybit is lagging by 50ms, still has old lower prices: Bid 100, Ask 101
    detector.update_price("Bybit", bid=100.0, ask=101.0, timestamp=current_time - 50)

    signal = detector.detect_arbitrage()

    assert signal is not None, "Expected an arbitrage signal"
    assert signal["action_buy"] == "Bybit", f"Expected to BUY on Bybit, got {signal['action_buy']}"
    assert signal["action_sell"] == "Binance" or signal["action_sell"] == "OKX", f"Expected to SELL on Binance or OKX, got {signal['action_sell']}"

    # We specifically want to check if Binance works as Sell side since prompt mentioned it
    # If profit is same, detector picks the first one it loops over. Let's make Binance slightly better.
    detector.update_price("Binance", bid=105.5, ask=106.5, timestamp=current_time)

    signal = detector.detect_arbitrage()
    assert signal["action_buy"] == "Bybit"
    assert signal["action_sell"] == "Binance"

    profit = signal["sell_price"] - signal["buy_price"]
    assert profit > 0
    assert signal["buy_timestamp"] == current_time - 50
    assert signal["sell_timestamp"] == current_time

    # Ensuring 50ms lag
    assert signal["sell_timestamp"] - signal["buy_timestamp"] == 50

import collections
import random

class VWAPSignal:
    def __init__(self, window=100, buy_deviation=-0.003, sell_deviation=0.003):
        self.window = window
        self.buy_deviation = buy_deviation
        self.sell_deviation = sell_deviation
        self.trades = collections.deque(maxlen=window)

    def add_trade(self, price: float, volume: float):
        self.trades.append((price, volume))

    def calculate_vwap(self) -> float:
        if not self.trades:
            return 0.0

        total_price_volume = sum(p * v for p, v in self.trades)
        total_volume = sum(v for p, v in self.trades)

        if total_volume == 0:
            return 0.0

        return total_price_volume / total_volume

    def get_signal(self, current_price: float) -> tuple:
        vwap = self.calculate_vwap()
        if vwap == 0.0:
            return ("HOLD", 0.0)

        deviation = (current_price - vwap) / vwap

        if deviation <= self.buy_deviation:
            return ("BUY", deviation)
        elif deviation >= self.sell_deviation:
            return ("SELL", deviation)
        else:
            return ("HOLD", deviation)

    def simulate_from_ohlcv(self, candles: list) -> list:
        results = []
        for candle in candles:
            timestamp, _, _, _, close, volume = candle
            self.add_trade(close, volume)
            signal, deviation = self.get_signal(close)
            results.append((timestamp, close, signal, deviation))
        return results

if __name__ == '__main__':
    # Generate 150 fake trades with random price around 58000 and random volume
    signal = VWAPSignal()
    for i in range(150):
        # random price around 58000 (e.g. within 1%)
        price = 58000 * (1 + random.uniform(-0.01, 0.01))
        # random volume between 0.1 and 5.0
        volume = random.uniform(0.1, 5.0)

        signal.add_trade(price, volume)
        sig, dev = signal.get_signal(price)

        if sig in ("BUY", "SELL"):
            print(f"Trade {i+1}: Price={price:.2f}, Signal={sig}, Deviation={dev:.4%}")

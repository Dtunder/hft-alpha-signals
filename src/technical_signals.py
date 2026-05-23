import collections
import math
import statistics
import random

class RSISignal:
    def __init__(self, period=14, overbought=70, oversold=30):
        self.gains = collections.deque(maxlen=period)
        self.losses = collections.deque(maxlen=period)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def update(self, price_change: float):
        if price_change > 0:
            self.gains.append(price_change)
            self.losses.append(0.0)
        elif price_change < 0:
            self.gains.append(0.0)
            self.losses.append(abs(price_change))
        else:
            self.gains.append(0.0)
            self.losses.append(0.0)

    def get_rsi(self) -> float:
        avg_gain = statistics.mean(self.gains) if self.gains else 0.0
        avg_loss = statistics.mean(self.losses) if self.losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def get_signal(self) -> str:
        rsi = self.get_rsi()
        if rsi < self.oversold:
            return "BUY"
        if rsi > self.overbought:
            return "SELL"
        return "HOLD"


class MACDSignal:
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.fast_ema = None
        self.slow_ema = None
        self.signal_line = None
        self.prev_macd = None
        self.prev_signal = None
        self.current_macd = None

    def _calculate_ema(self, price: float, prev_ema: float, n: int) -> float:
        if prev_ema is None:
            return price
        k = 2.0 / (n + 1)
        return price * k + prev_ema * (1.0 - k)

    def update(self, price: float):
        self.fast_ema = self._calculate_ema(price, self.fast_ema, self.fast)
        self.slow_ema = self._calculate_ema(price, self.slow_ema, self.slow)

        current_macd = self.fast_ema - self.slow_ema

        self.prev_macd = self.current_macd
        self.current_macd = current_macd

        self.prev_signal = self.signal_line
        self.signal_line = self._calculate_ema(self.current_macd, self.signal_line, self.signal)

    def get_signal(self) -> tuple:
        macd = self.current_macd
        sig = self.signal_line

        if macd is None or sig is None or self.prev_macd is None or self.prev_signal is None:
            return ("HOLD", macd if macd is not None else 0.0, sig if sig is not None else 0.0)

        if self.prev_macd < self.prev_signal and macd > sig:
            return ("BUY", macd, sig)
        if self.prev_macd > self.prev_signal and macd < sig:
            return ("SELL", macd, sig)

        return ("HOLD", macd, sig)


class BollingerSignal:
    def __init__(self, period=20, num_std=2.0):
        self.prices = collections.deque(maxlen=period)
        self.num_std = num_std

    def update(self, price: float):
        self.prices.append(price)

    def get_signal(self, current_price: float) -> tuple:
        if len(self.prices) < 2:
            return ("HOLD", current_price, current_price, current_price)

        middle = statistics.mean(self.prices)
        try:
            stdev = statistics.stdev(self.prices)
        except statistics.StatisticsError:
            stdev = 0.0

        upper = middle + self.num_std * stdev
        lower = middle - self.num_std * stdev

        if current_price < lower:
            return ("BUY", upper, lower, middle)
        if current_price > upper:
            return ("SELL", upper, lower, middle)
        return ("HOLD", upper, lower, middle)

    def get_bandwidth(self) -> float:
        if len(self.prices) < 2:
            return 0.0

        middle = statistics.mean(self.prices)
        try:
            stdev = statistics.stdev(self.prices)
        except statistics.StatisticsError:
            stdev = 0.0

        upper = middle + self.num_std * stdev
        lower = middle - self.num_std * stdev

        if middle == 0:
            return 0.0
        return (upper - lower) / middle


class TechnicalEnsemble:
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, bb_period=20):
        self.rsi = RSISignal(period=rsi_period)
        self.macd = MACDSignal(fast=macd_fast, slow=macd_slow, signal=9)
        self.bb = BollingerSignal(period=bb_period)
        self.prev_price = None

    def update(self, price: float):
        if self.prev_price is not None:
            price_change = price - self.prev_price
        else:
            price_change = 0.0

        self.rsi.update(price_change)
        self.macd.update(price)
        self.bb.update(price)

        self.prev_price = price

    def get_signal(self) -> dict:
        if self.prev_price is None:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "rsi": {"signal": "HOLD", "value": 100.0},
                "macd": {"signal": "HOLD", "macd": 0.0, "signal_line": 0.0},
                "bollinger": {"signal": "HOLD", "upper": 0.0, "lower": 0.0, "bandwidth": 0.0}
            }

        rsi_sig = self.rsi.get_signal()
        macd_sig, macd_line, sig_line = self.macd.get_signal()
        bb_sig, bb_upper, bb_lower, bb_middle = self.bb.get_signal(self.prev_price)

        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        votes[rsi_sig] += 1
        votes[macd_sig] += 1
        votes[bb_sig] += 1

        majority = "HOLD"
        max_votes = 0
        for k, v in votes.items():
            if v > max_votes:
                max_votes = v
                majority = k

        if max_votes == 1:
            majority = "HOLD"

        return {
            "signal": majority,
            "confidence": max_votes / 3.0,
            "rsi": {
                "signal": rsi_sig,
                "value": self.rsi.get_rsi()
            },
            "macd": {
                "signal": macd_sig,
                "macd": macd_line,
                "signal_line": sig_line
            },
            "bollinger": {
                "signal": bb_sig,
                "upper": bb_upper,
                "lower": bb_lower,
                "bandwidth": self.bb.get_bandwidth()
            }
        }


if __name__ == "__main__":
    ensemble = TechnicalEnsemble()
    price = 58000.0
    mu = 0.0001
    sigma = 0.01

    buy_count = 0
    sell_count = 0
    hold_count = 0

    # Use fixed seed for reproducible test or just random
    random.seed(42)

    for i in range(200):
        # Geometric Brownian Motion step
        W = random.gauss(0, 1)
        price = price * math.exp((mu - 0.5 * sigma**2) + sigma * W)

        ensemble.update(price)

        # Every 10 prices, get signal and print
        if i % 10 == 0:
            sig_info = ensemble.get_signal()
            print(f"Step {i:3d} | Price: {price:8.2f} | Signal: {sig_info['signal']:4s} | Confidence: {sig_info['confidence']:.2f}")

        # Count all signals generated over 200 steps
        sig_info = ensemble.get_signal()
        if sig_info["signal"] == "BUY":
            buy_count += 1
        elif sig_info["signal"] == "SELL":
            sell_count += 1
        else:
            hold_count += 1

    print("-" * 50)
    print("Signal Summary:")
    print(f"Total BUY signals : {buy_count}")
    print(f"Total SELL signals: {sell_count}")
    print(f"Total HOLD signals: {hold_count}")

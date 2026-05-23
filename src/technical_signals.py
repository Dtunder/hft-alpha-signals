import collections
import statistics
import math
import random

class RSISignal:
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.gains = collections.deque(maxlen=period)
        self.losses = collections.deque(maxlen=period)

    def update(self, price_change: float):
        if price_change > 0:
            self.gains.append(price_change)
            self.losses.append(0.0)
        else:
            self.gains.append(0.0)
            self.losses.append(abs(price_change))

    def get_rsi(self) -> float:
        avg_gain = statistics.mean(self.gains) if self.gains else 0
        avg_loss = statistics.mean(self.losses) if self.losses else 0

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
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal

        self.fast_ema = None
        self.slow_ema = None
        self.signal_line = None

        self.prev_macd = None
        self.prev_signal = None
        self.current_macd = None

    def _calc_ema(self, price, prev_ema, period):
        if prev_ema is None:
            return price
        k = 2.0 / (period + 1.0)
        return price * k + prev_ema * (1.0 - k)

    def update(self, price: float):
        self.fast_ema = self._calc_ema(price, self.fast_ema, self.fast_period)
        self.slow_ema = self._calc_ema(price, self.slow_ema, self.slow_period)

        # Shift values
        self.prev_macd = self.current_macd
        self.prev_signal = self.signal_line

        self.current_macd = self.fast_ema - self.slow_ema
        self.signal_line = self._calc_ema(self.current_macd, self.signal_line, self.signal_period)

    def get_signal(self) -> tuple:
        # Default to HOLD initially
        sig = "HOLD"

        if self.prev_macd is not None and self.prev_signal is not None and self.signal_line is not None:
            if self.prev_macd < self.prev_signal and self.current_macd > self.signal_line:
                sig = "BUY"
            elif self.prev_macd > self.prev_signal and self.current_macd < self.signal_line:
                sig = "SELL"

        return (sig, self.current_macd if self.current_macd is not None else 0.0,
                self.signal_line if self.signal_line is not None else 0.0)

class BollingerSignal:
    def __init__(self, period=20, num_std=2.0):
        self.period = period
        self.num_std = num_std
        self.prices = collections.deque(maxlen=period)

        self.middle = None
        self.upper = None
        self.lower = None

    def update(self, price: float):
        self.prices.append(price)
        if len(self.prices) >= 2:
            self.middle = statistics.mean(self.prices)
            std_dev = statistics.stdev(self.prices)
            self.upper = self.middle + self.num_std * std_dev
            self.lower = self.middle - self.num_std * std_dev
        else:
            self.middle = price
            self.upper = price
            self.lower = price

    def get_signal(self, current_price: float) -> tuple:
        sig = "HOLD"
        if self.lower is not None and self.upper is not None:
            if current_price < self.lower:
                sig = "BUY"
            elif current_price > self.upper:
                sig = "SELL"

        return (sig,
                self.upper if self.upper is not None else current_price,
                self.lower if self.lower is not None else current_price,
                self.middle if self.middle is not None else current_price)

    def get_bandwidth(self) -> float:
        if self.middle is not None and self.middle != 0 and self.upper is not None and self.lower is not None:
            return (self.upper - self.lower) / self.middle
        return 0.0

class TechnicalEnsemble:
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, bb_period=20):
        self.rsi = RSISignal(period=rsi_period)
        self.macd = MACDSignal(fast=macd_fast, slow=macd_slow)
        self.bb = BollingerSignal(period=bb_period)
        self.prev_price = None
        self.current_price = None

    def update(self, price: float):
        if self.prev_price is not None:
            price_change = price - self.prev_price
            self.rsi.update(price_change)

        self.macd.update(price)
        self.bb.update(price)

        self.prev_price = price
        self.current_price = price

    def get_signal(self) -> dict:
        rsi_sig = self.rsi.get_signal()
        macd_sig, macd_val, macd_signal_line = self.macd.get_signal()
        bb_sig, bb_upper, bb_lower, bb_middle = self.bb.get_signal(self.current_price if self.current_price else 0)

        votes = [rsi_sig, macd_sig, bb_sig]
        counts = collections.Counter(votes)

        # Majority vote
        majority_signal = "HOLD"
        if counts["BUY"] >= 2:
            majority_signal = "BUY"
        elif counts["SELL"] >= 2:
            majority_signal = "SELL"

        winning_votes = counts[majority_signal]

        return {
            "signal": majority_signal,
            "confidence": winning_votes / 3.0,
            "rsi": {"signal": rsi_sig, "value": self.rsi.get_rsi()},
            "macd": {"signal": macd_sig, "macd": macd_val, "signal_line": macd_signal_line},
            "bollinger": {"signal": bb_sig, "upper": bb_upper, "lower": bb_lower, "bandwidth": self.bb.get_bandwidth()}
        }

if __name__ == "__main__":
    ensemble = TechnicalEnsemble()
    price = 58000.0

    total_buy = 0
    total_sell = 0
    total_hold = 0

    # Simulate geometric brownian motion
    mu = 0.0001
    sigma = 0.01

    for i in range(1, 201):
        # GBM step
        shock = random.gauss(0, 1)
        price = price * math.exp((mu - 0.5 * sigma**2) + sigma * shock)

        ensemble.update(price)

        if i % 10 == 0:
            sig_dict = ensemble.get_signal()
            print(f"Step {i}: Price {price:.2f} | Signal: {sig_dict['signal']} (Conf: {sig_dict['confidence']:.2f})")

            if sig_dict["signal"] == "BUY":
                total_buy += 1
            elif sig_dict["signal"] == "SELL":
                total_sell += 1
            else:
                total_hold += 1

    print("\n--- Summary ---")
    print(f"Total BUY signals: {total_buy}")
    print(f"Total SELL signals: {total_sell}")
    print(f"Total HOLD signals: {total_hold}")

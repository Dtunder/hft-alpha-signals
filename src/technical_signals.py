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
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal
        self.fast_ema = None
        self.slow_ema = None
        self.signal_line = None
        self.prev_macd = None
        self.prev_signal = None

    def _calc_ema(self, price, prev_ema, n):
        k = 2.0 / (n + 1)
        if prev_ema is None:
            return price
        return price * k + prev_ema * (1.0 - k)

    def update(self, price: float):
        if self.fast_ema is not None and self.slow_ema is not None:
            self.prev_macd = self.fast_ema - self.slow_ema
        if self.signal_line is not None:
            self.prev_signal = self.signal_line

        self.fast_ema = self._calc_ema(price, self.fast_ema, self.fast_period)
        self.slow_ema = self._calc_ema(price, self.slow_ema, self.slow_period)

        macd = self.fast_ema - self.slow_ema
        self.signal_line = self._calc_ema(macd, self.signal_line, self.signal_period)

    def get_signal(self) -> tuple:
        if self.fast_ema is None or self.slow_ema is None or self.signal_line is None:
            return ("HOLD", 0.0, 0.0)

        macd = self.fast_ema - self.slow_ema

        if self.prev_macd is not None and self.prev_signal is not None:
            if self.prev_macd < self.prev_signal and macd > self.signal_line:
                return ("BUY", macd, self.signal_line)
            if self.prev_macd > self.prev_signal and macd < self.signal_line:
                return ("SELL", macd, self.signal_line)

        return ("HOLD", macd, self.signal_line)


class BollingerSignal:
    def __init__(self, period=20, num_std=2.0):
        self.period = period
        self.num_std = num_std
        self.prices = collections.deque(maxlen=period)

    def update(self, price: float):
        self.prices.append(price)

    def get_signal(self, current_price: float) -> tuple:
        if len(self.prices) < 2:
            return ("HOLD", current_price, current_price, current_price)

        middle = statistics.mean(self.prices)
        stdev = statistics.stdev(self.prices)
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
        if middle == 0:
            return 0.0
        stdev = statistics.stdev(self.prices)
        upper = middle + self.num_std * stdev
        lower = middle - self.num_std * stdev
        return (upper - lower) / middle


class TechnicalEnsemble:
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, bb_period=20):
        self.rsi = RSISignal(period=rsi_period)
        self.macd = MACDSignal(fast=macd_fast, slow=macd_slow, signal=9)
        self.bb = BollingerSignal(period=bb_period)
        self.prev_price = None

    def update(self, price: float):
        if self.prev_price is not None:
            self.rsi.update(price - self.prev_price)
        self.macd.update(price)
        self.bb.update(price)
        self.prev_price = price

    def get_signal(self) -> dict:
        if self.prev_price is None:
            return {
                "signal": "HOLD", "confidence": 0.0,
                "rsi": {"signal": "HOLD", "value": 100.0},
                "macd": {"signal": "HOLD", "macd": 0.0, "signal_line": 0.0},
                "bollinger": {"signal": "HOLD", "upper": 0.0, "lower": 0.0, "bandwidth": 0.0},
            }

        rsi_sig = self.rsi.get_signal()
        rsi_val = self.rsi.get_rsi()
        macd_sig, macd_line, sig_line = self.macd.get_signal()
        bb_sig, bb_upper, bb_lower, _ = self.bb.get_signal(self.prev_price)
        bb_bw = self.bb.get_bandwidth()

        votes = [rsi_sig, macd_sig, bb_sig]
        buy_votes = votes.count("BUY")
        sell_votes = votes.count("SELL")
        hold_votes = votes.count("HOLD")

        if buy_votes > sell_votes and buy_votes > hold_votes:
            majority = "BUY"
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            majority = "SELL"
        else:
            majority = "HOLD"

        confidence = max(buy_votes, sell_votes, hold_votes) / 3.0

        return {
            "signal": majority,
            "confidence": confidence,
            "rsi": {"signal": rsi_sig, "value": rsi_val},
            "macd": {"signal": macd_sig, "macd": macd_line, "signal_line": sig_line},
            "bollinger": {"signal": bb_sig, "upper": bb_upper, "lower": bb_lower, "bandwidth": bb_bw},
        }


if __name__ == "__main__":
    ensemble = TechnicalEnsemble()
    price = 58000.0
    total_buys = total_sells = total_holds = 0
    drift, volatility = 0.0001, 0.01

    for i in range(1, 201):
        price = price * math.exp((drift - 0.5 * volatility**2) + volatility * random.gauss(0, 1))
        ensemble.update(price)
        signal_data = ensemble.get_signal()
        sig = signal_data["signal"]
        if sig == "BUY":
            total_buys += 1
        elif sig == "SELL":
            total_sells += 1
        else:
            total_holds += 1
        if i % 10 == 0:
            print(f"Step {i:3d}: Price={price:.2f}  Signal={sig:4s}  Confidence={signal_data['confidence']:.2f}")

    print(f"\nTotal — BUY: {total_buys}  SELL: {total_sells}  HOLD: {total_holds}")

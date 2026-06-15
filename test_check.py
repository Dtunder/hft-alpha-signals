import collections

class HFTAlphaSignals:
    def __init__(self, obi_threshold=0.70):
        self.obi_threshold = obi_threshold
        self.obi_history = collections.deque(maxlen=3)

    def analyze_order_book(self, bids, asks):
        total_bid_volume = sum(bid[1] for bid in bids)
        total_ask_volume = sum(ask[1] for ask in asks)
        denominator = total_bid_volume + total_ask_volume
        return (total_bid_volume - total_ask_volume) / denominator if denominator else 0.0

    def check_signals(self, bids, asks, funding_rate=0.0):
        obi = self.analyze_order_book(bids, asks)

        if obi >= self.obi_threshold:
            raw_signal = "BUY"
        elif obi <= -self.obi_threshold:
            raw_signal = "SELL"
        else:
            raw_signal = "HOLD"

        signal = raw_signal
        self.obi_history.append(obi)

        if signal in ["BUY", "SELL"]:
            if len(self.obi_history) < 3:
                signal = "HOLD"
            else:
                if signal == "BUY":
                    if not all(past_obi > 0 for past_obi in self.obi_history):
                        signal = "HOLD"
                    if funding_rate > 0.0001:
                        signal = "HOLD"
                elif signal == "SELL":
                    if not all(past_obi < 0 for past_obi in self.obi_history):
                        signal = "HOLD"

        return signal, obi

brain = HFTAlphaSignals()
print(brain.check_signals([[100, 100]], [[101, 1]])) # tick 1
print(brain.check_signals([[100, 100]], [[101, 1]])) # tick 2
print(brain.check_signals([[100, 100]], [[101, 1]])) # tick 3 (should BUY)

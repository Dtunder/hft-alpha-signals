import random
import time
from collections import deque

class HFTAlphaSignals:
    """
    Identifies high-frequency alpha signals based on Order Book Imbalance (OBI).
    If bids vastly outweigh asks, upward pressure is imminent.
    """
    def __init__(self, obi_threshold=0.75, momentum_window=3):
        self.obi_threshold = max(0.60, min(0.75, obi_threshold))
        self.momentum_window = momentum_window
        self.mid_price_history = deque(maxlen=momentum_window)
        print(f"[ALPHA] Brain initialized. Monitoring Order Book Imbalance (OBI) with threshold {self.obi_threshold:.2f}...")

    def update_price_history(self, bid_depth, ask_depth):
        if not bid_depth or not ask_depth:
            return

        best_bid = bid_depth[0][0]
        best_ask = ask_depth[0][0]
        mid_price = (best_bid + best_ask) / 2.0

        self.mid_price_history.append(mid_price)

    def check_momentum(self):
        """
        Checks if the recent mid-prices confirm the momentum.
        Returns:
          1 for upward momentum,
         -1 for downward momentum,
          0 for no clear momentum.
        """
        if len(self.mid_price_history) < self.momentum_window:
            return 0 # Not enough history

        is_upward = all(self.mid_price_history[i] > self.mid_price_history[i-1] for i in range(1, len(self.mid_price_history)))
        is_downward = all(self.mid_price_history[i] < self.mid_price_history[i-1] for i in range(1, len(self.mid_price_history)))

        if is_upward:
            return 1
        elif is_downward:
            return -1
        else:
            return 0

    def analyze_order_book(self, bid_depth, ask_depth):
        """
        Calculates OBI: (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        Ranges from -1 (total asks) to 1 (total bids).
        """
        total_bid_volume = sum(bid[1] for bid in bid_depth)
        total_ask_volume = sum(ask[1] for ask in ask_depth)
        
        denominator = total_bid_volume + total_ask_volume
        if denominator == 0:
            return 0.0
            
        obi = (total_bid_volume - total_ask_volume) / denominator
        return obi

    def check_signals(self, bid_depth, ask_depth):
        """
        Scans book depth and generates BUY/SELL triggers with momentum confirmation.
        """
        self.update_price_history(bid_depth, ask_depth)
        obi = self.analyze_order_book(bid_depth, ask_depth)
        momentum = self.check_momentum()
        
        # Absolute OBI signals strong directional pressure, but require momentum confirmation
        if obi >= self.obi_threshold:
            if momentum == 1:
                return "BUY", obi
            else:
                return "HOLD (Fake OBI Spike)", obi
        elif obi <= -self.obi_threshold:
            if momentum == -1:
                return "SELL", obi
            else:
                return "HOLD (Fake OBI Spike)", obi
        else:
            return "HOLD", obi

if __name__ == "__main__":
    brain = HFTAlphaSignals()
    
    # Simulate 5 ticks of live order book depth
    # Format: list of [price, volume]

    base_price = 58000
    for tick in range(1, 6):
        print(f"\n--- [TICK #{tick}] ---")
        
        # Randomly skew depth to create momentum signals
        bid_skew = random.uniform(10, 100)
        ask_skew = random.uniform(10, 100)
        
        # Simulate price momentum: price goes up by $5 every tick
        base_price += 5

        bids = [[base_price - i, random.uniform(1.0, 10.0) * bid_skew] for i in range(5)]
        asks = [[base_price + 1 + i, random.uniform(1.0, 10.0) * ask_skew] for i in range(5)]
        
        signal, OBI = brain.check_signals(bids, asks)
        print(f"Mid Price: {(bids[0][0] + asks[0][0]) / 2.0:.2f}")
        print(f"Bids Vol: {sum(b[1] for b in bids):.2f} | Asks Vol: {sum(a[1] for a in asks):.2f}")
        print(f"Calculated OBI: {OBI:+.4f}")
        
        if signal.startswith("BUY") or signal.startswith("SELL"):
            print(f"🔥 [SIGNAL DETECTED] Triggering HIGH-SPEED {signal} order (OBI: {OBI:+.4f})!")
        else:
            print(f"💤 [SIGNAL] {signal}")
            
        time.sleep(0.5)

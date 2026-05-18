import random
import time

class HFTAlphaSignals:
    """
    Identifies high-frequency alpha signals based on Order Book Imbalance (OBI).
    If bids vastly outweigh asks, upward pressure is imminent.
    """
    def __init__(self, obi_threshold=0.75):
        self.obi_threshold = obi_threshold
        print("[ALPHA] Brain initialized. Monitoring Order Book Imbalance (OBI)...")

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
        Scans book depth and generates BUY/SELL triggers.
        """
        obi = self.analyze_order_book(bid_depth, ask_depth)
        
        # Absolute OBI signals strong directional pressure
        if obi >= self.obi_threshold:
            return "BUY", obi
        elif obi <= -self.obi_threshold:
            return "SELL", obi
        else:
            return "HOLD", obi

if __name__ == "__main__":
    brain = HFTAlphaSignals()
    
    # Simulate 5 ticks of live order book depth
    # Format: list of [price, volume]
    for tick in range(1, 6):
        print(f"\n--- [TICK #{tick}] ---")
        
        # Randomly skew depth to create momentum signals
        bid_skew = random.uniform(10, 100)
        ask_skew = random.uniform(10, 100)
        
        bids = [[58000 - i, random.uniform(1.0, 10.0) * bid_skew] for i in range(5)]
        asks = [[58001 + i, random.uniform(1.0, 10.0) * ask_skew] for i in range(5)]
        
        signal, OBI = brain.check_signals(bids, asks)
        print(f"Bids Vol: {sum(b[1] for b in bids):.2f} | Asks Vol: {sum(a[1] for a in asks):.2f}")
        print(f"Calculated OBI: {OBI:+.4f}")
        
        if signal != "HOLD":
            print(f"🔥 [SIGNAL DETECTED] Triggering HIGH-SPEED {signal} order (OBI: {OBI:+.4f})!")
        else:
            print("💤 [SIGNAL] Equilibrium maintained. Holding.")
            
        time.sleep(0.5)

import random
import time
import math

class HFTAlphaSignals:
    """
    Identifies high-frequency alpha signals based on Order Book Imbalance (OBI).
    If bids vastly outweigh asks, upward pressure is imminent.
    """
    def __init__(self, obi_threshold=0.75):
        self.obi_threshold = obi_threshold
        self.prev_bids = None
        self.prev_asks = None
        self.kf_estimate = 0.0
        self.kf_error = 1.0
        self.kf_q = 1e-5
        self.kf_r = 1e-3
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

    def compute_ofi(self, bid_depth, ask_depth):
        """
        Computes Order Flow Imbalance across up to 10 depth levels based on changes
        in bid/ask volumes and prices between the previous tick and current tick.
        """
        if self.prev_bids is None or self.prev_asks is None:
            self.prev_bids = bid_depth
            self.prev_asks = ask_depth
            return 0.0

        net_ofi = 0.0
        levels = min(len(bid_depth), len(ask_depth), len(self.prev_bids), len(self.prev_asks), 10)

        for n in range(levels):
            # Bid OFI
            current_bid_price, current_bid_vol = bid_depth[n]
            prev_bid_price, prev_bid_vol = self.prev_bids[n]

            bid_ofi = 0.0
            if current_bid_price >= prev_bid_price:
                bid_ofi += current_bid_vol
            if current_bid_price <= prev_bid_price:
                bid_ofi -= prev_bid_vol

            # Ask OFI
            current_ask_price, current_ask_vol = ask_depth[n]
            prev_ask_price, prev_ask_vol = self.prev_asks[n]

            ask_ofi = 0.0
            if current_ask_price <= prev_ask_price:
                ask_ofi += current_ask_vol
            if current_ask_price >= prev_ask_price:
                ask_ofi -= prev_ask_vol

            net_ofi += (bid_ofi - ask_ofi)

        self.prev_bids = bid_depth
        self.prev_asks = ask_depth

        return net_ofi

    def _kalman_update(self, measurement):
        """
        Updates the 1D Kalman filter state with a new measurement and returns the smoothed estimate.
        """
        # Prediction update
        self.kf_error = self.kf_error + self.kf_q

        # Measurement update
        kalman_gain = self.kf_error / (self.kf_error + self.kf_r)
        self.kf_estimate = self.kf_estimate + kalman_gain * (measurement - self.kf_estimate)
        self.kf_error = (1 - kalman_gain) * self.kf_error

        return self.kf_estimate

    def check_signals(self, bid_depth, ask_depth):
        """
        Scans book depth and generates BUY/SELL triggers.
        """
        obi = self.analyze_order_book(bid_depth, ask_depth)
        
        raw_ofi = self.compute_ofi(bid_depth, ask_depth)
        smoothed_ofi = self._kalman_update(raw_ofi)

        norm_ofi = math.tanh(smoothed_ofi / 100.0)

        composite_pressure = (obi + norm_ofi) / 2.0

        # Composite pressure signals strong directional pressure
        if composite_pressure >= self.obi_threshold:
            return "BUY", composite_pressure
        elif composite_pressure <= -self.obi_threshold:
            return "SELL", composite_pressure
        else:
            return "HOLD", composite_pressure

if __name__ == "__main__":
    brain = HFTAlphaSignals()
    
    # Simulate 5 ticks of live order book depth
    # Format: list of [price, volume]
    for tick in range(1, 6):
        print(f"\n--- [TICK #{tick}] ---")
        
        # Randomly skew depth to create momentum signals
        bid_skew = random.uniform(10, 100)
        ask_skew = random.uniform(10, 100)
        
        bids = [[58000 - i, random.uniform(1.0, 10.0) * bid_skew] for i in range(10)]
        asks = [[58001 + i, random.uniform(1.0, 10.0) * ask_skew] for i in range(10)]
        
        signal, composite_pressure = brain.check_signals(bids, asks)
        print(f"Bids Vol: {sum(b[1] for b in bids):.2f} | Asks Vol: {sum(a[1] for a in asks):.2f}")
        print(f"Calculated Composite Pressure: {composite_pressure:+.4f}")
        
        if signal != "HOLD":
            print(f"🔥 [SIGNAL DETECTED] Triggering HIGH-SPEED {signal} order (Pressure: {composite_pressure:+.4f})!")
        else:
            print("💤 [SIGNAL] Equilibrium maintained. Holding.")
            
        time.sleep(0.5)

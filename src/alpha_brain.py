import random
import time
import os
import joblib
import torch
import numpy as np

# We need the model definition to load the state dict
from train_model import GoalConditionedTCN

class HFTAlphaSignals:
    """
    Identifies high-frequency alpha signals based on Order Book Imbalance (OBI).
    If bids vastly outweigh asks, upward pressure is imminent.
    Now upgraded to use a Goal-Conditioned TCN for multi-pair analysis.
    """
    def __init__(self, obi_threshold=0.75, model_path="brain/multi_alpha_model.zip"):
        self.obi_threshold = obi_threshold
        self.model = None
        self.seq_len = 10
        self.history = []

        print("[ALPHA] Brain initialized. Monitoring Order Book Imbalance (OBI)...")

        if os.path.exists(model_path):
            print(f"[ALPHA] Loading Goal-Conditioned TCN from {model_path}...")
            # We assume num_pairs = 2 and 1 goal -> input_size = 3
            self.model = GoalConditionedTCN(input_size=3, num_channels=[16, 32, 64])
            state_dict = joblib.load(model_path)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print("[ALPHA] No TCN model found, falling back to basic OBI thresholding.")

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

    def check_signals(self, bid_depth, ask_depth, pair2_bid_depth=None, pair2_ask_depth=None, goal=0.5):
        """
        Scans book depth and generates BUY/SELL triggers.
        Uses TCN if available, else falls back to basic threshold.
        """
        obi1 = self.analyze_order_book(bid_depth, ask_depth)
        obi2 = self.analyze_order_book(pair2_bid_depth or [], pair2_ask_depth or []) if pair2_bid_depth is not None else obi1
        
        if self.model:
            # Multi-pair TCN prediction
            self.history.append([obi1, obi2, goal])
            if len(self.history) > self.seq_len:
                self.history.pop(0)

            if len(self.history) == self.seq_len:
                x = torch.tensor([self.history], dtype=torch.float32)
                with torch.no_grad():
                    output = self.model(x)
                    pred = torch.argmax(output, dim=1).item()

                if pred == 1:
                    return "BUY", obi1
                elif pred == 2:
                    return "SELL", obi1
                else:
                    return "HOLD", obi1
            else:
                return "HOLD", obi1 # Accumulating history
        else:
            # Fallback
            if obi1 >= self.obi_threshold:
                return "BUY", obi1
            elif obi1 <= -self.obi_threshold:
                return "SELL", obi1
            else:
                return "HOLD", obi1

if __name__ == "__main__":
    brain = HFTAlphaSignals()
    
    # Simulate 15 ticks of live order book depth to fill TCN sequence
    # Format: list of [price, volume]
    for tick in range(1, 16):
        print(f"\n--- [TICK #{tick}] ---")
        
        # Randomly skew depth to create momentum signals for Pair 1
        bid_skew1 = random.uniform(10, 100)
        ask_skew1 = random.uniform(10, 100)

        bids1 = [[58000 - i, random.uniform(1.0, 10.0) * bid_skew1] for i in range(5)]
        asks1 = [[58001 + i, random.uniform(1.0, 10.0) * ask_skew1] for i in range(5)]

        # Randomly skew depth for Pair 2 (e.g., correlated asset)
        bid_skew2 = random.uniform(10, 100)
        ask_skew2 = random.uniform(10, 100)
        
        bids2 = [[3000 - i, random.uniform(1.0, 10.0) * bid_skew2] for i in range(5)]
        asks2 = [[3001 + i, random.uniform(1.0, 10.0) * ask_skew2] for i in range(5)]
        
        signal, OBI = brain.check_signals(bids1, asks1, bids2, asks2, goal=0.8)
        print(f"Pair1 Bids Vol: {sum(b[1] for b in bids1):.2f} | Asks Vol: {sum(a[1] for a in asks1):.2f}")
        print(f"Pair1 Calculated OBI: {OBI:+.4f}")
        
        if signal != "HOLD":
            print(f"🔥 [SIGNAL DETECTED] Triggering HIGH-SPEED {signal} order (OBI: {OBI:+.4f})!")
        else:
            print("💤 [SIGNAL] Equilibrium maintained. Holding.")
            
        time.sleep(0.5)

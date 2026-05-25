import math
import collections
import random

class LiquidationHeatmap:
    def __init__(self):
        # Positions grouped by (side, leverage) -> list of (entry_price, size)
        self.positions = collections.defaultdict(list)
        self.leverages = [5, 10, 20, 50, 100]

    def update(self, price: float, open_interest: float, ls_ratio: float, funding_rate: float):
        """
        Estimate new positions based on Open Interest, Long/Short Ratio, and Funding Rates.
        """
        # A simple approximation: funding rate tilts the aggressiveness of positions
        # Positive funding -> longs pay shorts, implies longs are majority/aggressive

        long_oi = open_interest * (ls_ratio / (1 + ls_ratio))
        short_oi = open_interest / (1 + ls_ratio)

        # Funding rate effect: increases weight on the crowded side
        long_weight = 1.0 + max(0, funding_rate * 100)
        short_weight = 1.0 + max(0, -funding_rate * 100)

        # Distribute new volume over different leverages
        # In reality, higher leverage has less volume, we simulate with inverse weights
        total_lev_weight = sum(1.0 / lev for lev in self.leverages)

        for lev in self.leverages:
            lev_weight = (1.0 / lev) / total_lev_weight

            # Simulated incoming volume for this tick
            new_long = (long_oi * 0.01) * lev_weight * long_weight
            new_short = (short_oi * 0.01) * lev_weight * short_weight

            self.positions[('long', lev)].append((price, new_long))
            self.positions[('short', lev)].append((price, new_short))

        # Clean up positions that are already liquidated
        self._cleanup_liquidated(price)

    def _cleanup_liquidated(self, current_price: float):
        """Remove positions that would have been liquidated by the current price."""
        for side, lev in list(self.positions.keys()):
            active_positions = []
            for entry_price, size in self.positions[(side, lev)]:
                if side == 'long':
                    liq_price = entry_price * (1 - 1.0 / lev)
                    if current_price > liq_price:
                        active_positions.append((entry_price, size))
                else: # short
                    liq_price = entry_price * (1 + 1.0 / lev)
                    if current_price < liq_price:
                        active_positions.append((entry_price, size))
            self.positions[(side, lev)] = active_positions

    def _get_liquidation_clusters(self, current_price: float, bucket_size: float = 10.0):
        """Group liquidations into price buckets."""
        long_liqs = collections.defaultdict(float)
        short_liqs = collections.defaultdict(float)

        for (side, lev), pos_list in self.positions.items():
            for entry_price, size in pos_list:
                if side == 'long':
                    liq_price = entry_price * (1 - 1.0 / lev)
                    bucket = math.floor(liq_price / bucket_size) * bucket_size
                    long_liqs[bucket] += size
                else:
                    liq_price = entry_price * (1 + 1.0 / lev)
                    bucket = math.floor(liq_price / bucket_size) * bucket_size
                    short_liqs[bucket] += size

        return long_liqs, short_liqs

    def get_signal(self, current_price: float):
        """
        Returns (signal, confidence).
        BUY if heavy short liquidations are clustered slightly above current price.
        SELL if long liquidations are clustered slightly below.
        """
        long_liqs, short_liqs = self._get_liquidation_clusters(current_price, bucket_size=5.0)

        # Define ranges for "slightly above" and "slightly below"
        short_sq_start = current_price
        short_sq_end = current_price * 1.05

        long_liq_start = current_price * 0.95
        long_liq_end = current_price

        short_vol_above = sum(vol for p, vol in short_liqs.items() if short_sq_start <= p <= short_sq_end)
        long_vol_below = sum(vol for p, vol in long_liqs.items() if long_liq_start <= p <= long_liq_end)

        if short_vol_above > long_vol_below * 1.5 and short_vol_above > 100:
            confidence = min(100.0, (short_vol_above / (long_vol_below + 1.0)) * 10)
            return "BUY", round(confidence, 2)
        elif long_vol_below > short_vol_above * 1.5 and long_vol_below > 100:
            confidence = min(100.0, (long_vol_below / (short_vol_above + 1.0)) * 10)
            return "SELL", round(confidence, 2)

        return "HOLD", 0.0

    def print_matrix(self, current_price: float):
        long_liqs, short_liqs = self._get_liquidation_clusters(current_price, bucket_size=5.0)
        print(f"\n--- Liquidation Matrix at Price {current_price:.2f} ---")

        print("Short Liquidations (Above Price):")
        short_keys = sorted([k for k in short_liqs.keys() if k >= current_price])
        if not short_keys:
            print("  None")
        for p in short_keys[:5]:
            print(f"  Price ~{p:.2f} : Volume {short_liqs[p]:.2f}")

        print("Long Liquidations (Below Price):")
        long_keys = sorted([k for k in long_liqs.keys() if k <= current_price], reverse=True)
        if not long_keys:
            print("  None")
        for p in long_keys[:5]:
            print(f"  Price ~{p:.2f} : Volume {long_liqs[p]:.2f}")
        print("-----------------------------------------")


if __name__ == '__main__':
    heatmap = LiquidationHeatmap()

    print("Simulating 20-tick market sequence...")
    base_price = 50000.0
    open_interest = 1000000.0

    for tick in range(1, 21):
        # Simulate price movement
        price = base_price + random.uniform(-100, 100)

        # Simulate market metrics
        ls_ratio = random.uniform(0.8, 1.2)
        funding_rate = random.uniform(-0.001, 0.001)
        open_interest += random.uniform(-5000, 10000)

        heatmap.update(price, open_interest, ls_ratio, funding_rate)

        signal, confidence = heatmap.get_signal(price)
        print(f"Tick {tick:02d} | Price: {price:.2f} | Signal: {signal} (Conf: {confidence}%)")

        # Print matrix every 5 ticks
        if tick % 5 == 0:
            heatmap.print_matrix(price)

        base_price = price # random walk

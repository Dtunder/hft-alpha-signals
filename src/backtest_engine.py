import json
import torch
import numpy as np
import os
from src.multi_asset_brain import MultiAssetBrain

class BacktestEngine:
    def __init__(self, num_assets=10, ticks=10000, window_size=100):
        self.num_assets = num_assets
        self.ticks = ticks
        self.window_size = window_size
        self.brain = MultiAssetBrain(num_assets=num_assets, window_size=window_size)

        # Simulate price paths (Random Walk with correlated assets)
        self.prices = self._simulate_prices()

    def _simulate_prices(self):
        print("Simulating high-frequency tick data...")
        # Create correlated price movements
        base_noise = np.random.randn(self.ticks, 1)
        asset_noise = np.random.randn(self.ticks, self.num_assets) * 0.5

        # Adding correlation (Asset 0 and 1 will be highly correlated)
        returns = base_noise * 0.5 + asset_noise
        returns[:, 1] = returns[:, 0] * 0.8 + np.random.randn(self.ticks) * 0.1

        # Cumulative sum to simulate prices
        prices = np.cumsum(returns, axis=0) + 100.0
        return torch.tensor(prices, dtype=torch.float32)

    def run(self):
        print("Starting backtest engine...")

        results = {
            "total_ticks": self.ticks,
            "latencies_us": [],
            "predictions": [],
            "actual_returns": [],
            "profit": 0.0,
            "win_rate": 0.0,
            "trades": 0
        }

        wins = 0
        threshold = 2.0  # Breakout score threshold to trigger trade

        # For simplicity, if we predict a breakout for asset i at tick t,
        # we check the return of asset i from t to t+5
        lookahead = 5

        for t in range(self.ticks - lookahead):
            current_prices = self.prices[t]

            corr, breakouts, latency = self.brain.update(current_prices)

            if corr is None:
                continue

            results["latencies_us"].append(latency)

            # Find the strongest breakout signal
            best_asset = torch.argmax(breakouts).item()
            best_score = breakouts[best_asset].item()

            if best_score > threshold:
                # Predict positive breakout for best_asset
                future_return = self.prices[t + lookahead, best_asset] - self.prices[t, best_asset]
                future_return = future_return.item()

                results["trades"] += 1

                if future_return > 0:
                    wins += 1
                    results["profit"] += future_return
                else:
                    results["profit"] += future_return # Loss

                results["predictions"].append({
                    "tick": t,
                    "asset": best_asset,
                    "score": best_score,
                    "actual_return": future_return
                })

        if results["trades"] > 0:
            results["win_rate"] = wins / results["trades"]

        warmup = min(10, max(0, len(results["latencies_us"]) - 1))
        avg_latency = np.mean(results["latencies_us"][warmup:]) if len(results["latencies_us"]) > warmup else 0.0
        max_latency = np.max(results["latencies_us"]) if len(results["latencies_us"]) > 0 else 0.0

        # Save Report
        os.makedirs("logs", exist_ok=True)
        report = {
            "metrics": {
                "total_trades": results["trades"],
                "win_rate": results["win_rate"],
                "total_profit": results["profit"],
                "average_latency_us": avg_latency,
                "max_latency_us": max_latency
            }
        }

        with open("logs/backtest_report.json", "w") as f:
            json.dump(report, f, indent=4)

        print(f"Backtest complete. Trades: {results['trades']}, Win Rate: {results['win_rate']:.2%}")
        print(f"Average Latency: {avg_latency:.2f} µs")
        print("Report saved to logs/backtest_report.json")

if __name__ == "__main__":
    engine = BacktestEngine(num_assets=10, ticks=5000, window_size=50)
    engine.run()

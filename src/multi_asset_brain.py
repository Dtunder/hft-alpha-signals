import torch
import torch.nn as nn
import time
import os

class MultiAssetBrainCore(nn.Module):
    """
    Core tensor computations for Multi-Asset Brain.
    Optimized and reduced for torch.jit.trace to achieve sub-200us latency.
    """
    def __init__(self, num_assets=10):
        super().__init__()
        self.num_assets = num_assets

    def forward(self, prices):
        # prices shape: (window_size, num_assets)

        # 1. Real-time correlation matrix
        mean = torch.mean(prices, dim=0, keepdim=True)
        centered = prices - mean
        var = torch.sum(centered ** 2, dim=0, keepdim=True)
        std = torch.sqrt(var) + 1e-8

        cov = torch.matmul(centered.T, centered)
        corr = cov / torch.matmul(std.T, std)

        # 2. Cointegration Vectors / Spreads (leading/lagging)
        # Using z-scores (normalized prices) to extract spreads
        norm_prices = centered / std

        # Current normalized prices
        current_norm = norm_prices[-1]

        # Spread matrix: current_spreads[i, j] = norm_i - norm_j
        current_spreads = current_norm.unsqueeze(1) - current_norm.unsqueeze(0)

        # 3. Predict local price breakouts
        # If asset i is lagging highly correlated asset j, asset i has upward breakout potential.
        # breakout_score_i = sum_j ( -current_spreads[i, j] * corr[i, j] )
        breakout_scores = torch.sum(-current_spreads * corr, dim=1)

        return corr, breakout_scores

class MultiAssetBrain:
    def __init__(self, num_assets=10, window_size=100):
        self.num_assets = num_assets
        self.window_size = window_size

        self.core = MultiAssetBrainCore(num_assets=num_assets)

        # JIT trace for ultra-low latency execution
        dummy_input = torch.randn(window_size, num_assets)
        self.traced_core = torch.jit.trace(self.core, dummy_input)

        # Save traced model to brain/ directory as per conventions
        os.makedirs("brain", exist_ok=True)
        torch.jit.save(self.traced_core, "brain/multi_asset_core_traced.pt")

        self.history = torch.zeros((window_size, num_assets))
        self.step = 0

    def update(self, new_prices):
        """
        Tick update at millisecond resolution.
        """
        # Shift history and add new tick
        self.history = torch.roll(self.history, shifts=-1, dims=0)

        if isinstance(new_prices, torch.Tensor):
            self.history[-1] = new_prices
        else:
            self.history[-1] = torch.tensor(new_prices, dtype=torch.float32)

        self.step += 1

        if self.step < self.window_size:
            return None, None, 0.0

        # Inference with precise latency measurement
        start_time = time.perf_counter_ns()
        with torch.no_grad():
            corr, breakouts = self.traced_core(self.history)
        end_time = time.perf_counter_ns()

        latency_us = (end_time - start_time) / 1000.0

        return corr, breakouts, latency_us

if __name__ == "__main__":
    # Smoke test
    brain = MultiAssetBrain(num_assets=10, window_size=50)
    print("Initializing Multi-Asset Brain (JIT Traced)...")
    for _ in range(55):
        prices = torch.randn(10).tolist()
        corr, breakouts, lat = brain.update(prices)
        if corr is not None:
            print(f"Inference Latency: {lat:.2f} µs | Breakout Scores: {breakouts.numpy()}")

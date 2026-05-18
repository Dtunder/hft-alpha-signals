# 🧠 HFT Alpha Signals Detector
*High-Speed Market Inefficiency Arbitrage Brain*

> [!NOTE]
> This module detects microsecond price anomalies and momentum signals. It is inspired by the algorithmic secrets of the high-frequency trading bot that turned $0.50$ into $\$50,000$ using order book imbalance (OBI) metrics.

## 📈 Alpha Methodology
- **Order Book Imbalance (OBI):** Analyzes the ratio of bid vs. ask depth volume. A massive bid imbalance indicates immediate upward pressure.
- **Statistical Arbitrage:** Monitors divergence between correlated trading pairs to trigger instant mean-reversion trades.
- **Fast Execution Trigger:** Generates buy/sell actions instantly when OBI thresholds are breached.

---

## ⚡ Execution Instructions
To test the signal detection engine:
```bash
python src/alpha_brain.py
```

import urllib.request
import json

class SpreadSignal:
    def __init__(self, min_spread_pct=0.05):
        self.min_spread_pct = min_spread_pct

    def fetch_best_price(self, exchange: str, symbol: str) -> dict:
        try:
            if exchange == "binance":
                url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    return {"bid": float(data["bidPrice"]), "ask": float(data["askPrice"])}
            elif exchange == "bybit":
                url = f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    ticker = data["result"]["list"][0]
                    return {"bid": float(ticker["bid1Price"]), "ask": float(ticker["ask1Price"])}
            else:
                return {"bid": 0.0, "ask": 0.0}
        except Exception as e:
            return {"bid": 0.0, "ask": 0.0}

    def get_signal(self, symbol_binance="BTCUSDT", symbol_bybit="BTCUSDT") -> dict:
        binance_data = self.fetch_best_price("binance", symbol_binance)
        bybit_data = self.fetch_best_price("bybit", symbol_bybit)

        binance_ask = binance_data["ask"]
        bybit_ask = bybit_data["ask"]

        if binance_ask == 0.0 or bybit_ask == 0.0:
            spread_pct = 0.0
            reverse_spread = 0.0
        else:
            spread_pct = (bybit_ask - binance_ask) / binance_ask * 100
            reverse_spread = (binance_ask - bybit_ask) / bybit_ask * 100

        if spread_pct > self.min_spread_pct:
            return {
                "signal": "ARB",
                "buy_on": "binance",
                "sell_on": "bybit",
                "spread_pct": spread_pct,
                "binance": binance_data,
                "bybit": bybit_data
            }
        elif reverse_spread > self.min_spread_pct:
            return {
                "signal": "ARB",
                "buy_on": "bybit",
                "sell_on": "binance",
                "spread_pct": reverse_spread,
                "binance": binance_data,
                "bybit": bybit_data
            }
        else:
            return {
                "signal": "NO_ARB",
                "spread_pct": max(spread_pct, reverse_spread),
                "binance": binance_data,
                "bybit": bybit_data
            }

if __name__ == "__main__":
    signal_generator = SpreadSignal(min_spread_pct=0.05)

    print("Testing BTC Arbitrage Signal:")
    btc_signal = signal_generator.get_signal(symbol_binance="BTCUSDT", symbol_bybit="BTCUSDT")
    print(json.dumps(btc_signal, indent=2))

    print("\nTesting ETH Arbitrage Signal:")
    eth_signal = signal_generator.get_signal(symbol_binance="ETHUSDT", symbol_bybit="ETHUSDT")
    print(json.dumps(eth_signal, indent=2))

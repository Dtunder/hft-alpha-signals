import urllib.request
import urllib.error
import json

class FundingRateSignal:
    def __init__(self, long_threshold=-0.0005, short_threshold=0.0010):
        """
        long_threshold: funding rate below this -> LONG bias (longs are paid)
        short_threshold: funding rate above this -> SHORT bias (shorts are paid)
        """
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def fetch_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        """
        Calls Binance public REST endpoint and returns the funding rate.
        On any error, returns 0.0 and prints a warning.
        """
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data and isinstance(data, list) and len(data) > 0:
                    return float(data[0].get("fundingRate", 0.0))
        except urllib.error.HTTPError as e:
            print(f"[WARNING] HTTP Error fetching funding rate for {symbol}: {e}")
        except urllib.error.URLError as e:
            print(f"[WARNING] URL Error fetching funding rate for {symbol}: {e}")
        except Exception as e:
            print(f"[WARNING] Unexpected error fetching funding rate for {symbol}: {e}")
        return 0.0

    def get_signal(self, symbol: str = "BTCUSDT") -> tuple:
        """
        Returns a tuple of (signal, rate).
        signal can be "LONG", "SHORT", or "NEUTRAL".
        """
        rate = self.fetch_funding_rate(symbol)
        if rate < self.long_threshold:
            return ("LONG", rate)
        elif rate > self.short_threshold:
            return ("SHORT", rate)
        else:
            return ("NEUTRAL", rate)

    def get_signal_for_symbols(self, symbols: list) -> dict:
        """
        Calls get_signal for each symbol, returns dict {symbol: (signal, rate)}
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_signal(symbol)
        return results

if __name__ == "__main__":
    signal_module = FundingRateSignal()
    symbols_to_test = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    print("--- Binance Perpetual Futures Funding Rate Signals ---")
    results = signal_module.get_signal_for_symbols(symbols_to_test)

    for symbol, (signal, rate) in results.items():
        print(f"Symbol: {symbol:<10} | Signal: {signal:<7} | Rate: {rate:.6f}")

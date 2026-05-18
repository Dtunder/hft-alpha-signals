class CrossExchangeArbitrageDetector:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.prices = {
            "Binance": {"bid": 0.0, "ask": 0.0, "timestamp": 0},
            "Bybit": {"bid": 0.0, "ask": 0.0, "timestamp": 0},
            "OKX": {"bid": 0.0, "ask": 0.0, "timestamp": 0},
        }

    def update_price(self, exchange, bid, ask, timestamp):
        """
        Update the current orderbook top for a given exchange.
        """
        if exchange in self.prices:
            self.prices[exchange] = {"bid": bid, "ask": ask, "timestamp": timestamp}
        else:
            raise ValueError(f"Unknown exchange: {exchange}")

    def detect_arbitrage(self):
        """
        Detects the most profitable cross-exchange arbitrage opportunity.
        Returns a dictionary with the signal details or None if no arbitrage is found.
        """
        signals = []
        exchanges = list(self.prices.keys())

        for i in range(len(exchanges)):
            for j in range(len(exchanges)):
                if i != j:
                    buy_exchange = exchanges[i]
                    sell_exchange = exchanges[j]

                    ask_price = self.prices[buy_exchange]["ask"]
                    bid_price = self.prices[sell_exchange]["bid"]

                    if ask_price == 0.0 or bid_price == 0.0:
                        continue

                    # Arbitrage condition: buy at ask on buy_exchange, sell at bid on sell_exchange
                    profit_margin = bid_price - ask_price
                    if profit_margin > self.threshold:
                        signals.append({
                            "action_buy": buy_exchange,
                            "action_sell": sell_exchange,
                            "buy_price": ask_price,
                            "sell_price": bid_price,
                            "profit": profit_margin,
                            "buy_timestamp": self.prices[buy_exchange]["timestamp"],
                            "sell_timestamp": self.prices[sell_exchange]["timestamp"],
                        })

        # Return the best signal based on profit margin
        if signals:
            signals.sort(key=lambda x: x["profit"], reverse=True)
            return signals[0]

        return None

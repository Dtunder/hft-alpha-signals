class LiquidationHeatmap:
    def __init__(self, leverage_tiers=None):
        if leverage_tiers is None:
            leverage_tiers = [10, 25, 50, 100]
        self.leverage_tiers = leverage_tiers
        self.current_price = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0

    def update(self, price: float, position_size: float, entry_price: float):
        self.current_price = price
        self.position_size = position_size
        self.entry_price = entry_price

    def get_liquidation_levels(self) -> dict:
        levels = {}
        if self.entry_price == 0 or self.position_size == 0:
            return levels

        for leverage in self.leverage_tiers:
            if self.position_size > 0:
                liq_price = self.entry_price * (1 - 1 / leverage)
            else:
                liq_price = self.entry_price * (1 + 1 / leverage)
            levels[leverage] = liq_price
        return levels

    def nearest_liquidation(self, current_price: float) -> dict:
        levels = self.get_liquidation_levels()
        if not levels:
            return {}

        nearest = {}
        min_dist = float('inf')

        for leverage, liq_price in levels.items():
            if current_price == 0:
                continue
            dist_pct = abs(current_price - liq_price) / current_price
            if dist_pct < min_dist:
                min_dist = dist_pct
                nearest = {
                    "leverage": leverage,
                    "liq_price": liq_price,
                    "distance_pct": min_dist
                }
        return nearest

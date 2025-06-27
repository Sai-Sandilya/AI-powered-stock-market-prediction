class WatchlistManager:
    def __init__(self):
        self.watchlists = {}
    def add_ticker(self, user_id: str, symbol: str):
        self.watchlists.setdefault(user_id, set()).add(symbol)
    def get_watchlist(self, user_id: str) -> list:
        return list(self.watchlists.get(user_id, []))

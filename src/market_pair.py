from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class MarketPair:
    """Matched market pair across two venues."""
    kalshi_ticker: str
    kalshi_end_time: datetime
    polymarket_slug: str
    polymarket_end_time: datetime
    match_confidence: float  # 0.0-1.0 fuzzy match score

    def expiration_diff_seconds(self) -> int:
        """Calculate absolute expiration time difference in seconds."""
        return abs(int((self.kalshi_end_time - self.polymarket_end_time).total_seconds()))

    def is_valid_pair(self, max_expiration_diff: int = 120) -> bool:
        """Check if pair has acceptable expiration time difference."""
        return self.expiration_diff_seconds() <= max_expiration_diff

    def __repr__(self) -> str:
        diff = self.expiration_diff_seconds()
        return f"Pair({self.kalshi_ticker}/{self.polymarket_slug}, diff={diff}s, conf={self.match_confidence:.2f})"

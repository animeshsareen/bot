"""Canonical equal-hedge quantity model for cross-venue arbitrage.

This module defines the core position and quantity model enforcing strict
equality across Polymarket and Kalshi venues.

Invariants:
- DONE_EQUAL: qty_poly == qty_kalshi == common_qty (successful hedge)
- FLAT_FORCED: qty_poly == qty_kalshi == 0 (emergency exit)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class PositionOutcome(Enum):
    """Terminal states for a position lifecycle."""

    DONE_EQUAL = auto()  # Successfully hedged: qty_poly == qty_kalshi == common_qty
    FLAT_FORCED = auto()  # Emergency flatten: qty_poly == qty_kalshi == 0


# Alias for convenience
Outcome = PositionOutcome


@dataclass
class PositionState:
    """Tracks position quantities across both venues.

    Enforces the equal-hedge invariant: both venues must always have
    equal quantities when in a terminal state.

    Attributes:
        target_qty: Single integer governing intended position size at both venues.
        qty_poly: Current quantity held at Polymarket.
        qty_kalshi: Current quantity held at Kalshi.
        outcome: Terminal state once reconciliation completes.
    """

    target_qty: int = 0
    qty_poly: int = 0
    qty_kalshi: int = 0
    outcome: Optional[PositionOutcome] = None

    def __post_init__(self) -> None:
        if self.target_qty < 0:
            raise ValueError("target_qty must be non-negative")
        if self.qty_poly < 0:
            raise ValueError("qty_poly must be non-negative")
        if self.qty_kalshi < 0:
            raise ValueError("qty_kalshi must be non-negative")

    @property
    def common_qty(self) -> int:
        """The minimum quantity held at both venues (hedged amount)."""
        return min(self.qty_poly, self.qty_kalshi)

    @property
    def is_balanced(self) -> bool:
        """True if both venues have equal quantities."""
        return self.qty_poly == self.qty_kalshi

    @property
    def is_flat(self) -> bool:
        """True if no position at either venue."""
        return self.qty_poly == 0 and self.qty_kalshi == 0

    @property
    def short_leg(self) -> Optional[str]:
        """Identifies which venue has fewer contracts, or None if balanced."""
        if self.qty_poly < self.qty_kalshi:
            return "poly"
        elif self.qty_kalshi < self.qty_poly:
            return "kalshi"
        return None

    @property
    def imbalance(self) -> int:
        """Absolute difference between venue quantities."""
        return abs(self.qty_poly - self.qty_kalshi)

    def fill_poly(self, qty: int) -> "PositionState":
        """Record a fill at Polymarket. Returns new state (immutable)."""
        if qty < 0:
            raise ValueError("fill qty must be non-negative")
        return PositionState(
            target_qty=self.target_qty,
            qty_poly=self.qty_poly + qty,
            qty_kalshi=self.qty_kalshi,
            outcome=self.outcome,
        )

    def fill_kalshi(self, qty: int) -> "PositionState":
        """Record a fill at Kalshi. Returns new state (immutable)."""
        if qty < 0:
            raise ValueError("fill qty must be non-negative")
        return PositionState(
            target_qty=self.target_qty,
            qty_poly=self.qty_poly,
            qty_kalshi=self.qty_kalshi + qty,
            outcome=self.outcome,
        )

    def reduce_poly(self, qty: int) -> "PositionState":
        """Reduce position at Polymarket (for flattening). Returns new state."""
        if qty < 0:
            raise ValueError("reduce qty must be non-negative")
        new_qty = max(0, self.qty_poly - qty)
        return PositionState(
            target_qty=self.target_qty,
            qty_poly=new_qty,
            qty_kalshi=self.qty_kalshi,
            outcome=self.outcome,
        )

    def reduce_kalshi(self, qty: int) -> "PositionState":
        """Reduce position at Kalshi (for flattening). Returns new state."""
        if qty < 0:
            raise ValueError("reduce qty must be non-negative")
        new_qty = max(0, self.qty_kalshi - qty)
        return PositionState(
            target_qty=self.target_qty,
            qty_poly=self.qty_poly,
            qty_kalshi=new_qty,
            outcome=self.outcome,
        )

    def reconcile(self) -> "PositionState":
        """Attempt to mark position as DONE_EQUAL if balanced with non-zero qty.

        Idempotent: calling multiple times has no additional effect.

        Returns:
            New PositionState with outcome set if invariant holds.
        """
        if self.outcome is not None:
            return self  # Already finalized

        if self.is_balanced and not self.is_flat:
            return PositionState(
                target_qty=self.target_qty,
                qty_poly=self.qty_poly,
                qty_kalshi=self.qty_kalshi,
                outcome=PositionOutcome.DONE_EQUAL,
            )
        return self

    def flatten(self) -> "PositionState":
        """Force both venues to zero and mark FLAT_FORCED.

        Idempotent: calling multiple times has no additional effect.

        Returns:
            New PositionState with both quantities at 0 and FLAT_FORCED outcome.
        """
        if self.outcome == PositionOutcome.FLAT_FORCED and self.is_flat:
            return self  # Already flat

        return PositionState(
            target_qty=0,
            qty_poly=0,
            qty_kalshi=0,
            outcome=PositionOutcome.FLAT_FORCED,
        )

    def validate_done_equal(self) -> bool:
        """Validate the DONE_EQUAL invariant holds.

        Returns:
            True if outcome is DONE_EQUAL and qty_poly == qty_kalshi == common_qty > 0
        """
        if self.outcome != PositionOutcome.DONE_EQUAL:
            return False
        return self.is_balanced and self.common_qty > 0

    def validate_flat_forced(self) -> bool:
        """Validate the FLAT_FORCED invariant holds.

        Returns:
            True if outcome is FLAT_FORCED and qty_poly == qty_kalshi == 0
        """
        if self.outcome != PositionOutcome.FLAT_FORCED:
            return False
        return self.is_flat


def create_position(target_qty: int) -> PositionState:
    """Factory function to create a new position with target quantity.

    Args:
        target_qty: Desired position size at both venues.

    Returns:
        Fresh PositionState with target set and zero fills.
    """
    return PositionState(target_qty=target_qty, qty_poly=0, qty_kalshi=0)

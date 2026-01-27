"""Tests for the canonical equal-hedge quantity model (US-001)."""

import pytest

from src.position import PositionOutcome, PositionState, create_position


class TestPositionStateBasics:
    """Basic PositionState creation and properties."""

    def test_create_position_factory(self):
        """create_position returns fresh state with target set."""
        pos = create_position(10)
        assert pos.target_qty == 10
        assert pos.qty_poly == 0
        assert pos.qty_kalshi == 0
        assert pos.outcome is None

    def test_single_target_qty_governs_both_venues(self):
        """Single integer target_qty governs both venues."""
        pos = PositionState(target_qty=15, qty_poly=0, qty_kalshi=0)
        assert pos.target_qty == 15

    def test_position_state_tracks_qty_explicitly(self):
        """PositionState tracks qty_poly and qty_kalshi explicitly."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=7)
        assert pos.qty_poly == 5
        assert pos.qty_kalshi == 7

    def test_common_qty_is_minimum(self):
        """common_qty is the minimum of both venues."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=7)
        assert pos.common_qty == 5

        pos2 = PositionState(target_qty=10, qty_poly=8, qty_kalshi=3)
        assert pos2.common_qty == 3

    def test_is_balanced_when_equal(self):
        """is_balanced returns True when both venues have equal qty."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        assert pos.is_balanced is True

        pos2 = PositionState(target_qty=10, qty_poly=5, qty_kalshi=6)
        assert pos2.is_balanced is False

    def test_is_flat_when_both_zero(self):
        """is_flat returns True when both venues are at zero."""
        pos = PositionState(target_qty=10, qty_poly=0, qty_kalshi=0)
        assert pos.is_flat is True

        pos2 = PositionState(target_qty=10, qty_poly=1, qty_kalshi=0)
        assert pos2.is_flat is False

    def test_short_leg_identification(self):
        """short_leg identifies the venue with fewer contracts."""
        pos = PositionState(target_qty=10, qty_poly=3, qty_kalshi=5)
        assert pos.short_leg == "poly"

        pos2 = PositionState(target_qty=10, qty_poly=7, qty_kalshi=4)
        assert pos2.short_leg == "kalshi"

        pos3 = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        assert pos3.short_leg is None

    def test_imbalance_calculation(self):
        """imbalance returns absolute difference."""
        pos = PositionState(target_qty=10, qty_poly=3, qty_kalshi=8)
        assert pos.imbalance == 5


class TestPositionStateFills:
    """Fill operations are immutable and accumulate correctly."""

    def test_fill_poly_immutable(self):
        """fill_poly returns new state without modifying original."""
        pos = create_position(10)
        new_pos = pos.fill_poly(5)

        assert pos.qty_poly == 0  # Original unchanged
        assert new_pos.qty_poly == 5
        assert new_pos.qty_kalshi == 0

    def test_fill_kalshi_immutable(self):
        """fill_kalshi returns new state without modifying original."""
        pos = create_position(10)
        new_pos = pos.fill_kalshi(7)

        assert pos.qty_kalshi == 0  # Original unchanged
        assert new_pos.qty_kalshi == 7
        assert new_pos.qty_poly == 0

    def test_multiple_fills_accumulate(self):
        """Multiple fills accumulate correctly."""
        pos = create_position(10)
        pos = pos.fill_poly(3)
        pos = pos.fill_poly(2)
        pos = pos.fill_kalshi(4)
        pos = pos.fill_kalshi(1)

        assert pos.qty_poly == 5
        assert pos.qty_kalshi == 5
        assert pos.is_balanced

    def test_fill_negative_raises(self):
        """Negative fill quantities raise ValueError."""
        pos = create_position(10)
        with pytest.raises(ValueError, match="non-negative"):
            pos.fill_poly(-1)
        with pytest.raises(ValueError, match="non-negative"):
            pos.fill_kalshi(-1)


class TestPositionStateReduces:
    """Reduce operations for flattening positions."""

    def test_reduce_poly_immutable(self):
        """reduce_poly returns new state without modifying original."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        new_pos = pos.reduce_poly(3)

        assert pos.qty_poly == 5  # Original unchanged
        assert new_pos.qty_poly == 2
        assert new_pos.qty_kalshi == 5

    def test_reduce_kalshi_immutable(self):
        """reduce_kalshi returns new state without modifying original."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        new_pos = pos.reduce_kalshi(2)

        assert pos.qty_kalshi == 5  # Original unchanged
        assert new_pos.qty_kalshi == 3
        assert new_pos.qty_poly == 5

    def test_reduce_clamps_to_zero(self):
        """Reducing more than available clamps to zero."""
        pos = PositionState(target_qty=10, qty_poly=3, qty_kalshi=3)
        new_pos = pos.reduce_poly(10)
        assert new_pos.qty_poly == 0

        new_pos2 = pos.reduce_kalshi(100)
        assert new_pos2.qty_kalshi == 0

    def test_reduce_negative_raises(self):
        """Negative reduce quantities raise ValueError."""
        pos = create_position(10)
        with pytest.raises(ValueError, match="non-negative"):
            pos.reduce_poly(-1)
        with pytest.raises(ValueError, match="non-negative"):
            pos.reduce_kalshi(-1)


class TestDoneEqualInvariant:
    """DONE_EQUAL invariant: qty_poly == qty_kalshi == common_qty."""

    def test_reconcile_sets_done_equal_when_balanced(self):
        """reconcile sets DONE_EQUAL when both venues have equal non-zero qty."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        reconciled = pos.reconcile()

        assert reconciled.outcome == PositionOutcome.DONE_EQUAL
        assert reconciled.validate_done_equal() is True

    def test_reconcile_does_not_set_when_imbalanced(self):
        """reconcile does not set outcome when venues are imbalanced."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=7)
        reconciled = pos.reconcile()

        assert reconciled.outcome is None

    def test_reconcile_does_not_set_when_flat(self):
        """reconcile does not set DONE_EQUAL when flat (zero qty)."""
        pos = PositionState(target_qty=10, qty_poly=0, qty_kalshi=0)
        reconciled = pos.reconcile()

        assert reconciled.outcome is None

    def test_done_equal_invariant_holds(self):
        """DONE_EQUAL invariant: qty_poly == qty_kalshi == common_qty."""
        pos = PositionState(target_qty=10, qty_poly=7, qty_kalshi=7)
        pos = pos.reconcile()

        assert pos.validate_done_equal() is True
        assert pos.qty_poly == pos.qty_kalshi == pos.common_qty == 7


class TestFlatForcedInvariant:
    """FLAT_FORCED invariant: qty_poly == qty_kalshi == 0."""

    def test_flatten_sets_flat_forced(self):
        """flatten sets FLAT_FORCED and zeros both venues."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=3)
        flattened = pos.flatten()

        assert flattened.outcome == PositionOutcome.FLAT_FORCED
        assert flattened.qty_poly == 0
        assert flattened.qty_kalshi == 0
        assert flattened.validate_flat_forced() is True

    def test_flatten_from_imbalanced(self):
        """flatten works from any imbalanced state."""
        pos = PositionState(target_qty=10, qty_poly=8, qty_kalshi=2)
        flattened = pos.flatten()

        assert flattened.is_flat
        assert flattened.outcome == PositionOutcome.FLAT_FORCED

    def test_flat_forced_invariant_holds(self):
        """FLAT_FORCED invariant: qty_poly == qty_kalshi == 0."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        pos = pos.flatten()

        assert pos.validate_flat_forced() is True
        assert pos.qty_poly == pos.qty_kalshi == 0


class TestIdempotence:
    """Reconciliation and flattening are idempotent."""

    def test_reconcile_idempotent(self):
        """Calling reconcile multiple times has no additional effect."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        r1 = pos.reconcile()
        r2 = r1.reconcile()
        r3 = r2.reconcile()

        assert r1.outcome == r2.outcome == r3.outcome == PositionOutcome.DONE_EQUAL
        assert r1.qty_poly == r2.qty_poly == r3.qty_poly == 5
        assert r1.qty_kalshi == r2.qty_kalshi == r3.qty_kalshi == 5

    def test_flatten_idempotent(self):
        """Calling flatten multiple times has no additional effect."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=3)
        f1 = pos.flatten()
        f2 = f1.flatten()
        f3 = f2.flatten()

        assert f1.outcome == f2.outcome == f3.outcome == PositionOutcome.FLAT_FORCED
        assert f1.is_flat and f2.is_flat and f3.is_flat

    def test_reconcile_after_done_equal_is_idempotent(self):
        """reconcile on already DONE_EQUAL state returns same state."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        reconciled = pos.reconcile()
        again = reconciled.reconcile()

        assert reconciled is again  # Same object returned


class TestValidation:
    """Validation of position state constraints."""

    def test_negative_target_qty_raises(self):
        """Negative target_qty raises ValueError."""
        with pytest.raises(ValueError, match="target_qty"):
            PositionState(target_qty=-1, qty_poly=0, qty_kalshi=0)

    def test_negative_qty_poly_raises(self):
        """Negative qty_poly raises ValueError."""
        with pytest.raises(ValueError, match="qty_poly"):
            PositionState(target_qty=10, qty_poly=-1, qty_kalshi=0)

    def test_negative_qty_kalshi_raises(self):
        """Negative qty_kalshi raises ValueError."""
        with pytest.raises(ValueError, match="qty_kalshi"):
            PositionState(target_qty=10, qty_poly=0, qty_kalshi=-1)


class TestInvariantValidation:
    """Validation helpers for invariants."""

    def test_validate_done_equal_false_when_wrong_outcome(self):
        """validate_done_equal returns False when outcome is not DONE_EQUAL."""
        pos = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5)
        assert pos.validate_done_equal() is False  # No outcome set

        pos2 = pos.flatten()
        assert pos2.validate_done_equal() is False  # FLAT_FORCED, not DONE_EQUAL

    def test_validate_done_equal_false_when_imbalanced(self):
        """validate_done_equal returns False when quantities don't match."""
        pos = PositionState(
            target_qty=10,
            qty_poly=5,
            qty_kalshi=3,
            outcome=PositionOutcome.DONE_EQUAL,  # Manually set invalid
        )
        assert pos.validate_done_equal() is False

    def test_validate_flat_forced_false_when_wrong_outcome(self):
        """validate_flat_forced returns False when outcome is not FLAT_FORCED."""
        pos = PositionState(target_qty=10, qty_poly=0, qty_kalshi=0)
        assert pos.validate_flat_forced() is False  # No outcome set

        pos2 = PositionState(target_qty=10, qty_poly=5, qty_kalshi=5).reconcile()
        assert pos2.validate_flat_forced() is False  # DONE_EQUAL, not FLAT_FORCED

    def test_validate_flat_forced_false_when_not_flat(self):
        """validate_flat_forced returns False when not actually flat."""
        pos = PositionState(
            target_qty=10,
            qty_poly=5,
            qty_kalshi=0,
            outcome=PositionOutcome.FLAT_FORCED,  # Manually set invalid
        )
        assert pos.validate_flat_forced() is False

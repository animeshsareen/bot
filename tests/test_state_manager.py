"""Comprehensive tests for State Manager components.

Tests cover:
1. FSM type definitions and transitions
2. State manager atomic operations
3. Event validation and ordering
4. Event consumer coroutine
5. Integration flow
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

# Import components under test
from src.types import (
    OrderState, AttemptState, EventType,
    Order, Fill, Attempt, Inventory, Event, Snapshot,
    is_valid_order_transition, is_valid_attempt_transition,
    ORDER_TRANSITIONS, ATTEMPT_TRANSITIONS,
    TERMINAL_ORDER_STATES, TERMINAL_ATTEMPT_STATES,
)
from src.state import (
    StateManager, Transaction,
    InvalidTransitionError, InvariantViolationError, OutOfOrderEventError,
    InventoryBandError,
    create_state_manager, derive_client_order_id,
)
from src.event_validator import (
    EventValidator, BoundedEventQueue, LRUSet,
    ValidationError, MissingFieldError, OutOfOrderError, DuplicateEventError,
    create_validator, create_event_queue,
)
from src.state_consumer import (
    EventConsumer, EventHandlers, DeadLetterQueue, ConsumerState,
    create_consumer,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state_manager():
    """Create fresh StateManager for each test."""
    return create_state_manager(kalshi_balance=1000.0, poly_balance=1000.0)


@pytest.fixture
def validator():
    """Create EventValidator for each test."""
    return create_validator(max_event_age_ms=60000)


@pytest.fixture
def event_queue():
    """Create event queue for each test."""
    return create_event_queue(max_size=100)


@pytest.fixture
def consumer(state_manager, validator, event_queue):
    """Create EventConsumer for each test."""
    return create_consumer(state_manager, validator, event_queue)


# =============================================================================
# TYPES TESTS
# =============================================================================

class TestOrderState:
    """Tests for order state machine."""

    def test_valid_transitions_from_pending(self):
        """PENDING can transition to SENT or CANCELLED."""
        assert is_valid_order_transition(OrderState.PENDING, OrderState.SENT)
        assert is_valid_order_transition(OrderState.PENDING, OrderState.CANCELLED)
        assert not is_valid_order_transition(OrderState.PENDING, OrderState.FILLED)

    def test_valid_transitions_from_sent(self):
        """SENT can transition to ACKED, REJECTED, or EXPIRED."""
        assert is_valid_order_transition(OrderState.SENT, OrderState.ACKED)
        assert is_valid_order_transition(OrderState.SENT, OrderState.REJECTED)
        assert is_valid_order_transition(OrderState.SENT, OrderState.EXPIRED)
        assert not is_valid_order_transition(OrderState.SENT, OrderState.FILLED)

    def test_valid_transitions_from_acked(self):
        """ACKED can transition to PARTIAL, FILLED, CANCELLED, EXPIRED."""
        assert is_valid_order_transition(OrderState.ACKED, OrderState.PARTIAL)
        assert is_valid_order_transition(OrderState.ACKED, OrderState.FILLED)
        assert is_valid_order_transition(OrderState.ACKED, OrderState.CANCELLED)
        assert is_valid_order_transition(OrderState.ACKED, OrderState.EXPIRED)

    def test_valid_transitions_from_partial(self):
        """PARTIAL can transition to FILLED, CANCELLED, EXPIRED."""
        assert is_valid_order_transition(OrderState.PARTIAL, OrderState.FILLED)
        assert is_valid_order_transition(OrderState.PARTIAL, OrderState.CANCELLED)
        assert is_valid_order_transition(OrderState.PARTIAL, OrderState.EXPIRED)

    def test_terminal_states_have_no_transitions(self):
        """Terminal states cannot transition to anything."""
        for terminal in TERMINAL_ORDER_STATES:
            for target in OrderState:
                if target != terminal:
                    assert not is_valid_order_transition(terminal, target)

    def test_order_with_state_validates(self):
        """Order.with_state validates transitions."""
        order = Order(
            order_id="test_1",
            venue="kalshi",
            market_id="TEST-MKT",
            side="buy",
            price=0.50,
            quantity=10,
            state=OrderState.PENDING,
        )

        # Valid transition
        updated = order.with_state(OrderState.SENT)
        assert updated.state == OrderState.SENT

        # Invalid transition
        with pytest.raises(ValueError):
            order.with_state(OrderState.FILLED)

    def test_order_with_state_preserves_client_order_id(self):
        """Order.with_state preserves client_order_id through transitions."""
        client_id = "A-attempt_test-KALSHI-YES"
        order = Order(
            order_id="test_1",
            venue="kalshi",
            market_id="TEST-MKT",
            side="buy",
            price=0.50,
            quantity=10,
            state=OrderState.PENDING,
            client_order_id=client_id,
        )

        # Transition PENDING -> SENT
        sent = order.with_state(OrderState.SENT)
        assert sent.client_order_id == client_id

        # Transition SENT -> ACKED
        acked = sent.with_state(OrderState.ACKED)
        assert acked.client_order_id == client_id

        # Transition ACKED -> FILLED (with additional updates)
        filled = acked.with_state(OrderState.FILLED, filled_qty=10, avg_fill_price=0.50)
        assert filled.client_order_id == client_id
        assert filled.filled_qty == 10


class TestAttemptState:
    """Tests for attempt state machine."""

    def test_valid_transitions_from_idle(self):
        """IDLE can only transition to EXECUTING."""
        assert is_valid_attempt_transition(AttemptState.IDLE, AttemptState.EXECUTING)
        assert not is_valid_attempt_transition(AttemptState.IDLE, AttemptState.RECONCILING)

    def test_valid_transitions_from_executing(self):
        """EXECUTING can transition to RECONCILING, FLATTENING, or FAILED."""
        assert is_valid_attempt_transition(AttemptState.EXECUTING, AttemptState.RECONCILING)
        assert is_valid_attempt_transition(AttemptState.EXECUTING, AttemptState.FLATTENING)
        assert is_valid_attempt_transition(AttemptState.EXECUTING, AttemptState.FAILED)

    def test_valid_transitions_from_reconciling(self):
        """RECONCILING can transition to DONE_*, FLATTENING, or FAILED."""
        assert is_valid_attempt_transition(AttemptState.RECONCILING, AttemptState.DONE_BALANCED)
        assert is_valid_attempt_transition(AttemptState.RECONCILING, AttemptState.DONE_IMBALANCED)
        assert is_valid_attempt_transition(AttemptState.RECONCILING, AttemptState.FLATTENING)
        assert is_valid_attempt_transition(AttemptState.RECONCILING, AttemptState.FAILED)

    def test_terminal_states_can_reset_to_idle(self):
        """Terminal attempt states can reset to IDLE."""
        for terminal in TERMINAL_ATTEMPT_STATES:
            assert is_valid_attempt_transition(terminal, AttemptState.IDLE)

    def test_attempt_is_balanced(self):
        """Attempt.is_balanced checks fill equality."""
        attempt = Attempt(
            attempt_id="test_1",
            state=AttemptState.DONE_BALANCED,
            strategy_path=1,
            target_qty=10,
            kalshi_filled=10,
            poly_filled=10,
        )
        assert attempt.is_balanced

        unbalanced = Attempt(
            attempt_id="test_2",
            state=AttemptState.DONE_IMBALANCED,
            strategy_path=1,
            target_qty=10,
            kalshi_filled=10,
            poly_filled=5,
        )
        assert not unbalanced.is_balanced


class TestInventory:
    """Tests for Inventory dataclass."""

    def test_with_fill_buy(self):
        """with_fill updates position and balance for buys."""
        inv = Inventory(kalshi_balance=100.0, poly_balance=100.0)
        updated = inv.with_fill("kalshi", qty=10, price=0.50, side="buy")

        assert updated.kalshi_position == 10
        assert updated.kalshi_balance == 95.0  # 100 - (10 * 0.50)
        assert updated.poly_position == 0
        assert updated.poly_balance == 100.0

    def test_with_fill_sell(self):
        """with_fill updates position and balance for sells."""
        inv = Inventory(kalshi_position=10, kalshi_balance=100.0)
        updated = inv.with_fill("kalshi", qty=5, price=0.60, side="sell")

        assert updated.kalshi_position == 5  # 10 - 5
        assert updated.kalshi_balance == 103.0  # 100 + (5 * 0.60)

    def test_is_flat(self):
        """is_flat checks for zero positions."""
        flat = Inventory()
        assert flat.is_flat

        not_flat = Inventory(kalshi_position=1)
        assert not not_flat.is_flat


class TestSnapshot:
    """Tests for Snapshot dataclass with versioning."""

    def test_snapshot_default_values(self):
        """Snapshot has correct default values for book sequences."""
        snap = Snapshot()
        assert snap.book_seq_kalshi == 0
        assert snap.book_seq_poly_yes == 0
        assert snap.book_seq_poly_no == 0

    def test_snapshot_with_sequences(self):
        """Snapshot stores book sequence numbers."""
        snap = Snapshot(
            poly_yes_ask=0.52,
            poly_yes_bid=0.48,
            kalshi_yes_ask=0.53,
            kalshi_yes_bid=0.47,
            book_seq_kalshi=100,
            book_seq_poly_yes=200,
            book_seq_poly_no=150,
        )
        assert snap.book_seq_kalshi == 100
        assert snap.book_seq_poly_yes == 200
        assert snap.book_seq_poly_no == 150

    def test_snapshot_id_tuple(self):
        """snapshot_id returns correct tuple format."""
        now_ms = int(time.time() * 1000)
        snap = Snapshot(
            book_seq_kalshi=10,
            book_seq_poly_yes=20,
            book_seq_poly_no=30,
            snapshot_ms=now_ms,
        )
        expected_id = (10, 20, 30, now_ms)
        assert snap.snapshot_id == expected_id

    def test_snapshot_id_unique_per_sequence(self):
        """Different sequences produce different snapshot_ids."""
        snap1 = Snapshot(book_seq_kalshi=1, book_seq_poly_yes=2, book_seq_poly_no=3, snapshot_ms=1000)
        snap2 = Snapshot(book_seq_kalshi=1, book_seq_poly_yes=2, book_seq_poly_no=4, snapshot_ms=1000)  # Different poly_no
        snap3 = Snapshot(book_seq_kalshi=1, book_seq_poly_yes=2, book_seq_poly_no=3, snapshot_ms=1001)  # Different time

        assert snap1.snapshot_id != snap2.snapshot_id
        assert snap1.snapshot_id != snap3.snapshot_id
        assert snap2.snapshot_id != snap3.snapshot_id

    def test_is_stale_fresh_snapshot(self):
        """Fresh snapshot is not stale."""
        snap = Snapshot(snapshot_ms=int(time.time() * 1000))
        assert not snap.is_stale()
        assert not snap.is_stale(max_age_ms=250)

    def test_is_stale_old_snapshot(self):
        """Old snapshot is stale."""
        old_time = int(time.time() * 1000) - 500  # 500ms ago
        snap = Snapshot(snapshot_ms=old_time)
        assert snap.is_stale(max_age_ms=250)

    def test_is_stale_custom_threshold(self):
        """is_stale respects custom max_age_ms."""
        old_time = int(time.time() * 1000) - 100  # 100ms ago
        snap = Snapshot(snapshot_ms=old_time)

        # 100ms old, 250ms threshold -> not stale
        assert not snap.is_stale(max_age_ms=250)

        # 100ms old, 50ms threshold -> stale
        assert snap.is_stale(max_age_ms=50)

    def test_is_stale_default_threshold_250ms(self):
        """Default staleness threshold is 250ms."""
        # Exactly at threshold - not stale
        snap_at_threshold = Snapshot(snapshot_ms=int(time.time() * 1000) - 250)
        assert not snap_at_threshold.is_stale()

        # Just over threshold - stale
        snap_over_threshold = Snapshot(snapshot_ms=int(time.time() * 1000) - 251)
        assert snap_over_threshold.is_stale()

    def test_snapshot_immutable(self):
        """Snapshot is immutable (frozen dataclass)."""
        snap = Snapshot(book_seq_kalshi=10)
        with pytest.raises(Exception):  # FrozenInstanceError
            snap.book_seq_kalshi = 20


class TestSnapshotVersioning:
    """Tests for snapshot versioning in StateManager."""

    def test_state_manager_snapshot_with_sequences(self, state_manager):
        """StateManager.snapshot() accepts and stores sequence numbers."""
        snap = state_manager.snapshot(
            poly_yes_ask=0.52,
            poly_yes_bid=0.48,
            kalshi_yes_ask=0.53,
            kalshi_yes_bid=0.47,
            book_seq_kalshi=100,
            book_seq_poly_yes=200,
            book_seq_poly_no=150,
        )

        assert snap.book_seq_kalshi == 100
        assert snap.book_seq_poly_yes == 200
        assert snap.book_seq_poly_no == 150
        assert snap.poly_yes_ask == 0.52
        assert snap.kalshi_yes_ask == 0.53

    def test_state_manager_snapshot_default_sequences(self, state_manager):
        """StateManager.snapshot() defaults sequences to 0."""
        snap = state_manager.snapshot(
            poly_yes_ask=0.52,
            kalshi_yes_ask=0.53,
        )

        assert snap.book_seq_kalshi == 0
        assert snap.book_seq_poly_yes == 0
        assert snap.book_seq_poly_no == 0

    def test_state_manager_snapshot_includes_inventory(self, state_manager):
        """StateManager.snapshot() includes current inventory."""
        # Modify inventory via fill
        order = state_manager.create_order("kalshi", "TEST", "buy", 0.50, 10)
        state_manager.update_order_state(order.order_id, OrderState.SENT)
        state_manager.update_order_state(order.order_id, OrderState.ACKED)
        state_manager.record_fill(order.order_id, quantity=10, price=0.50)

        snap = state_manager.snapshot(book_seq_kalshi=5)

        assert snap.inventory.kalshi_position == 10
        assert snap.book_seq_kalshi == 5

    def test_snapshot_staleness_detection_for_executor(self, state_manager):
        """Snapshot staleness can be checked before execution."""
        # Simulate: arb loop creates snapshot
        snap = state_manager.snapshot(
            poly_yes_ask=0.52,
            kalshi_yes_ask=0.53,
            book_seq_kalshi=10,
            book_seq_poly_yes=20,
            book_seq_poly_no=15,
        )

        # Snapshot is fresh immediately
        assert not snap.is_stale()

        # Simulate passage of time (in real code, executor would check this)
        # Create an artificially old snapshot for testing
        old_snap = Snapshot(
            poly_yes_ask=0.52,
            kalshi_yes_ask=0.53,
            book_seq_kalshi=10,
            book_seq_poly_yes=20,
            book_seq_poly_no=15,
            snapshot_ms=int(time.time() * 1000) - 300,  # 300ms ago
        )

        assert old_snap.is_stale()  # Executor should reject this


# =============================================================================
# STATE MANAGER TESTS
# =============================================================================

class TestStateManager:
    """Tests for StateManager."""

    def test_create_order(self, state_manager):
        """create_order creates order in PENDING state."""
        order = state_manager.create_order(
            venue="kalshi",
            market_id="TEST-MKT",
            side="buy",
            price=0.50,
            quantity=10,
        )

        assert order.state == OrderState.PENDING
        assert order.venue == "kalshi"
        assert order.quantity == 10
        assert order.filled_qty == 0

    def test_create_order_with_leg_generates_client_order_id(self, state_manager):
        """create_order with attempt_id and leg generates client_order_id."""
        attempt = state_manager.create_attempt(strategy_path=1, target_qty=10)

        order = state_manager.create_order(
            venue="kalshi",
            market_id="TEST-MKT",
            side="buy",
            price=0.50,
            quantity=10,
            attempt_id=attempt.attempt_id,
            leg="YES",
        )

        # Format: A-{attempt_suffix}-{VENUE}-{LEG}
        attempt_suffix = attempt.attempt_id.replace("attempt_", "")
        expected_client_id = f"A-{attempt_suffix}-KALSHI-YES"

        assert order.client_order_id == expected_client_id
        assert order.state == OrderState.PENDING
        assert order.attempt_id == attempt.attempt_id

    def test_create_order_without_leg_uses_fallback_client_order_id(self, state_manager):
        """create_order without leg uses order_id as fallback client_order_id."""
        order = state_manager.create_order(
            venue="kalshi",
            market_id="TEST-MKT",
            side="buy",
            price=0.50,
            quantity=10,
        )

        # Fallback: client_order_id == order_id
        assert order.client_order_id == order.order_id

    def test_create_order_with_attempt_but_no_leg_uses_fallback(self, state_manager):
        """create_order with attempt_id but no leg uses fallback."""
        attempt = state_manager.create_attempt(strategy_path=1, target_qty=10)

        order = state_manager.create_order(
            venue="kalshi",
            market_id="TEST-MKT",
            side="buy",
            price=0.50,
            quantity=10,
            attempt_id=attempt.attempt_id,
            # leg not provided
        )

        # Fallback: client_order_id == order_id
        assert order.client_order_id == order.order_id
        assert order.attempt_id == attempt.attempt_id

    def test_client_order_id_preserved_through_state_transitions(self, state_manager):
        """client_order_id is preserved when order state changes."""
        attempt = state_manager.create_attempt(strategy_path=1, target_qty=10)

        order = state_manager.create_order(
            venue="poly",
            market_id="TEST-MKT",
            side="sell",
            price=0.45,
            quantity=5,
            attempt_id=attempt.attempt_id,
            leg="NO",
        )

        original_client_id = order.client_order_id

        # Transition through states
        state_manager.update_order_state(order.order_id, OrderState.SENT)
        updated = state_manager.update_order_state(order.order_id, OrderState.ACKED)

        assert updated.client_order_id == original_client_id
        assert updated.state == OrderState.ACKED

    def test_derive_client_order_id_helper(self):
        """derive_client_order_id generates correct format."""
        client_id = derive_client_order_id("attempt_abc12345", "kalshi", "YES")
        assert client_id == "A-abc12345-KALSHI-YES"

        client_id = derive_client_order_id("attempt_xyz789", "poly", "no")
        assert client_id == "A-xyz789-POLY-NO"

    def test_client_order_id_format_matches_derive_helper(self, state_manager):
        """create_order client_order_id matches derive_client_order_id helper."""
        attempt = state_manager.create_attempt(strategy_path=1, target_qty=10)

        order = state_manager.create_order(
            venue="kalshi",
            market_id="TEST-MKT",
            side="buy",
            price=0.50,
            quantity=10,
            attempt_id=attempt.attempt_id,
            leg="YES",
        )

        expected = derive_client_order_id(attempt.attempt_id, "kalshi", "YES")
        assert order.client_order_id == expected

    def test_create_order_invalid_quantity(self, state_manager):
        """create_order rejects non-positive quantity."""
        with pytest.raises(InvariantViolationError):
            state_manager.create_order(
                venue="kalshi",
                market_id="TEST",
                side="buy",
                price=0.50,
                quantity=0,
            )

    def test_update_order_state_valid(self, state_manager):
        """update_order_state allows valid transitions."""
        order = state_manager.create_order("kalshi", "TEST", "buy", 0.50, 10)

        updated = state_manager.update_order_state(order.order_id, OrderState.SENT)
        assert updated.state == OrderState.SENT

        updated = state_manager.update_order_state(order.order_id, OrderState.ACKED)
        assert updated.state == OrderState.ACKED

    def test_update_order_state_invalid(self, state_manager):
        """update_order_state rejects invalid transitions."""
        order = state_manager.create_order("kalshi", "TEST", "buy", 0.50, 10)

        with pytest.raises(InvalidTransitionError):
            state_manager.update_order_state(order.order_id, OrderState.FILLED)

    def test_record_fill(self, state_manager):
        """record_fill updates order, inventory, and creates fill."""
        order = state_manager.create_order("kalshi", "TEST", "buy", 0.50, 10)
        state_manager.update_order_state(order.order_id, OrderState.SENT)
        state_manager.update_order_state(order.order_id, OrderState.ACKED)

        fill = state_manager.record_fill(order.order_id, quantity=5, price=0.50)

        # Check fill
        assert fill.quantity == 5
        assert fill.price == 0.50

        # Check order updated
        updated_order = state_manager.get_order(order.order_id)
        assert updated_order.state == OrderState.PARTIAL
        assert updated_order.filled_qty == 5

        # Check inventory updated
        inv = state_manager.get_inventory()
        assert inv.kalshi_position == 5

    def test_record_fill_exceeds_quantity(self, state_manager):
        """record_fill rejects fill exceeding order quantity."""
        order = state_manager.create_order("kalshi", "TEST", "buy", 0.50, 10)
        state_manager.update_order_state(order.order_id, OrderState.SENT)
        state_manager.update_order_state(order.order_id, OrderState.ACKED)

        with pytest.raises(InvariantViolationError):
            state_manager.record_fill(order.order_id, quantity=15, price=0.50)

    def test_create_attempt(self, state_manager):
        """create_attempt creates attempt and tracks as active."""
        attempt = state_manager.create_attempt(strategy_path=1, target_qty=10)

        assert attempt.state == AttemptState.IDLE
        assert attempt.strategy_path == 1
        assert state_manager.get_active_attempt() == attempt

    def test_create_attempt_while_active(self, state_manager):
        """create_attempt rejects if another attempt is active."""
        state_manager.create_attempt(strategy_path=1, target_qty=10)

        with pytest.raises(InvariantViolationError):
            state_manager.create_attempt(strategy_path=2, target_qty=5)

    # =========================================================================
    # INVENTORY BAND CHECK TESTS
    # =========================================================================

    def test_can_start_attempt_flat_inventory(self, state_manager):
        """can_start_attempt returns True when inventory is flat."""
        assert state_manager.can_start_attempt()
        assert state_manager.can_start_attempt(max_delta=5)

    def test_can_start_attempt_within_band(self, state_manager):
        """can_start_attempt returns True when |net_position| <= max_delta."""
        # Simulate imbalanced fill: kalshi +3, poly 0 -> net = 3
        state_manager.set_inventory(Inventory(kalshi_position=3, poly_position=0))
        assert state_manager.can_start_attempt(max_delta=5)  # 3 <= 5
        assert state_manager.can_start_attempt(max_delta=3)  # 3 <= 3

    def test_can_start_attempt_exceeds_band(self, state_manager):
        """can_start_attempt returns False when |net_position| > max_delta."""
        # Simulate imbalanced fill: kalshi +6, poly 0 -> net = 6
        state_manager.set_inventory(Inventory(kalshi_position=6, poly_position=0))
        assert not state_manager.can_start_attempt(max_delta=5)  # 6 > 5

    def test_can_start_attempt_negative_position(self, state_manager):
        """can_start_attempt handles negative net positions."""
        # Simulate: kalshi 0, poly -7 -> net = -7
        state_manager.set_inventory(Inventory(kalshi_position=0, poly_position=-7))
        assert not state_manager.can_start_attempt(max_delta=5)  # |-7| > 5
        assert state_manager.can_start_attempt(max_delta=7)       # |-7| <= 7

    def test_create_attempt_checks_inventory_band(self, state_manager):
        """create_attempt raises InventoryBandError when inventory exceeds band."""
        # Set imbalanced inventory exceeding default band (5)
        state_manager.set_inventory(Inventory(kalshi_position=10, poly_position=0))

        with pytest.raises(InventoryBandError) as exc_info:
            state_manager.create_attempt(strategy_path=1, target_qty=5)

        assert "net_position=10" in str(exc_info.value)
        assert "exceeds band" in str(exc_info.value)

    def test_create_attempt_with_custom_band(self, state_manager):
        """create_attempt accepts custom max_inventory_delta."""
        state_manager.set_inventory(Inventory(kalshi_position=8, poly_position=0))

        # Should fail with default band (5)
        with pytest.raises(InventoryBandError):
            state_manager.create_attempt(strategy_path=1, target_qty=5)

        # Should succeed with larger band
        attempt = state_manager.create_attempt(
            strategy_path=1,
            target_qty=5,
            max_inventory_delta=10,
        )
        assert attempt.state == AttemptState.IDLE

    def test_create_attempt_inventory_band_checked_before_active_attempt(self, state_manager):
        """Inventory band check happens before active attempt check."""
        # Set inventory that exceeds band
        state_manager.set_inventory(Inventory(kalshi_position=20, poly_position=0))

        # Should raise InventoryBandError, not InvariantViolationError
        with pytest.raises(InventoryBandError):
            state_manager.create_attempt(strategy_path=1, target_qty=5)

    def test_default_inventory_band_is_5(self, state_manager):
        """Default inventory band is 5 shares."""
        assert StateManager.DEFAULT_INVENTORY_BAND == 5

        # Position of 5 should be allowed
        state_manager.set_inventory(Inventory(kalshi_position=5, poly_position=0))
        attempt = state_manager.create_attempt(strategy_path=1, target_qty=5)
        assert attempt is not None

    def test_transaction_rollback(self, state_manager):
        """Transaction rolls back on exception."""
        order = state_manager.create_order("kalshi", "TEST", "buy", 0.50, 10)
        initial_count = len(state_manager._orders)

        try:
            with state_manager.transaction():
                state_manager.update_order_state(order.order_id, OrderState.SENT)
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        # Should be rolled back to PENDING
        assert state_manager.get_order(order.order_id).state == OrderState.PENDING

    def test_sequence_tracking(self, state_manager):
        """check_sequence tracks and validates sequence numbers."""
        assert state_manager.check_sequence("kalshi", 1)
        assert state_manager.check_sequence("kalshi", 2)
        assert not state_manager.check_sequence("kalshi", 1)  # Out of order

    def test_check_invariants(self, state_manager):
        """check_invariants detects violations."""
        # Valid state should have no violations
        assert state_manager.check_invariants() == []


# =============================================================================
# EVENT VALIDATOR TESTS
# =============================================================================

class TestEventValidator:
    """Tests for EventValidator."""

    def test_valid_order_sent_event(self, validator):
        """Valid ORDER_SENT event passes validation."""
        event = Event(
            event_type=EventType.ORDER_SENT,
            sequence_num=1,
            venue="kalshi",
            order_id="order_1",
        )
        result = validator.validate(event)
        assert result.valid

    def test_missing_required_field(self, validator):
        """Event with missing required field fails."""
        event = Event(
            event_type=EventType.ORDER_SENT,
            sequence_num=1,
            venue="kalshi",
            # Missing order_id
        )
        result = validator.validate(event)
        assert not result.valid
        assert result.error_type == "missing_field"

    def test_invalid_venue(self, validator):
        """Event with invalid venue fails."""
        event = Event(
            event_type=EventType.ORDER_SENT,
            sequence_num=1,
            venue="invalid_venue",
            order_id="order_1",
        )
        result = validator.validate(event)
        assert not result.valid
        assert result.error_type == "invalid_field"

    def test_duplicate_detection(self, validator):
        """Duplicate events are rejected."""
        event = Event(
            event_type=EventType.ORDER_SENT,
            sequence_num=1,
            venue="kalshi",
            order_id="order_1",
        )

        result1 = validator.validate(event)
        assert result1.valid

        result2 = validator.validate(event)
        assert not result2.valid
        assert result2.error_type == "duplicate"

    def test_out_of_order_rejection(self, validator):
        """Out of order events are rejected."""
        event1 = Event(
            event_type=EventType.ORDER_SENT,
            sequence_num=5,
            venue="kalshi",
            order_id="order_1",
        )
        event2 = Event(
            event_type=EventType.ORDER_ACKED,
            sequence_num=3,  # Earlier sequence
            venue="kalshi",
            order_id="order_1",
            venue_order_id="venue_1",
        )

        validator.validate(event1)
        result = validator.validate(event2)
        assert not result.valid
        assert result.error_type == "out_of_order"

    def test_stale_event_rejection(self):
        """Very old events are rejected."""
        validator = EventValidator(max_event_age_ms=1000)
        old_event = Event(
            event_type=EventType.ORDER_SENT,
            sequence_num=1,
            timestamp_ms=int(time.time() * 1000) - 5000,  # 5 seconds ago
            venue="kalshi",
            order_id="order_1",
        )
        result = validator.validate(old_event)
        assert not result.valid
        assert result.error_type == "stale"


class TestLRUSet:
    """Tests for LRU duplicate cache."""

    def test_add_new_item(self):
        """Adding new item returns True."""
        lru = LRUSet(max_size=3)
        assert lru.add("a")
        assert "a" in lru

    def test_add_duplicate_returns_false(self):
        """Adding duplicate returns False."""
        lru = LRUSet(max_size=3)
        lru.add("a")
        assert not lru.add("a")

    def test_eviction(self):
        """Oldest items are evicted when over capacity."""
        lru = LRUSet(max_size=3)
        lru.add("a")
        lru.add("b")
        lru.add("c")
        lru.add("d")  # Should evict "a"

        assert "a" not in lru
        assert "d" in lru
        assert len(lru) == 3


class TestBoundedEventQueue:
    """Tests for event queue with backpressure."""

    def test_enqueue_dequeue(self, event_queue):
        """Basic enqueue/dequeue works."""
        event = Event(event_type=EventType.SHUTDOWN, sequence_num=1)
        event_queue.enqueue(event)
        result = event_queue.dequeue()
        assert result == event

    def test_backpressure_signaling(self):
        """Backpressure is signaled at high watermark."""
        queue = BoundedEventQueue(max_size=10, high_watermark_pct=0.5)

        for i in range(4):
            queue.enqueue(Event(event_type=EventType.SHUTDOWN, sequence_num=i))
        assert not queue.backpressure

        queue.enqueue(Event(event_type=EventType.SHUTDOWN, sequence_num=5))
        assert queue.backpressure

    def test_drop_oldest_policy(self):
        """Drop oldest policy evicts oldest when full."""
        queue = BoundedEventQueue(max_size=3, drop_policy="oldest")

        for i in range(5):
            queue.enqueue(Event(event_type=EventType.SHUTDOWN, sequence_num=i))

        # Should have events 2, 3, 4 (oldest 0, 1 dropped)
        assert queue.size == 3
        assert queue.dequeue().sequence_num == 2


# =============================================================================
# EVENT CONSUMER TESTS
# =============================================================================

class TestEventConsumer:
    """Tests for EventConsumer."""

    @pytest.mark.asyncio
    async def test_consumer_processes_events(self, consumer, state_manager, event_queue):
        """Consumer processes events from queue."""
        # Create an order first
        order = state_manager.create_order("kalshi", "TEST", "buy", 0.50, 10)

        # Enqueue ORDER_SENT event
        event = Event(
            event_type=EventType.ORDER_SENT,
            sequence_num=1,
            venue="kalshi",
            order_id=order.order_id,
        )
        event_queue.enqueue(event)

        # Enqueue shutdown to stop consumer
        shutdown = Event(event_type=EventType.SHUTDOWN, sequence_num=2)
        event_queue.enqueue(shutdown)

        # Run consumer
        await consumer.run()

        # Verify order state changed
        updated = state_manager.get_order(order.order_id)
        assert updated.state == OrderState.SENT

    @pytest.mark.asyncio
    async def test_consumer_handles_invalid_transition(self, consumer, state_manager, event_queue):
        """Consumer handles invalid transitions gracefully."""
        order = state_manager.create_order("kalshi", "TEST", "buy", 0.50, 10)

        # Try invalid transition (PENDING -> FILLED)
        event = Event(
            event_type=EventType.ORDER_FILL,
            sequence_num=1,
            venue="kalshi",
            order_id=order.order_id,
            fill_qty=5,
            fill_price=0.50,
        )
        event_queue.enqueue(event)
        event_queue.enqueue(Event(event_type=EventType.SHUTDOWN, sequence_num=2))

        await consumer.run()

        # Order should still be PENDING
        assert state_manager.get_order(order.order_id).state == OrderState.PENDING
        # Event should be dead-lettered after retries
        assert consumer.dead_letters.size > 0 or consumer.metrics.events_failed > 0

    @pytest.mark.asyncio
    async def test_consumer_graceful_shutdown(self, consumer, event_queue):
        """Consumer drains queue on shutdown."""
        # Add multiple events
        for i in range(5):
            event_queue.enqueue(Event(event_type=EventType.INVENTORY_UPDATE, sequence_num=i))
        event_queue.enqueue(Event(event_type=EventType.SHUTDOWN, sequence_num=10))

        await consumer.run()

        # Queue should be empty
        assert event_queue.size == 0
        assert consumer.state == ConsumerState.STOPPED


class TestDeadLetterQueue:
    """Tests for dead letter queue."""

    def test_add_and_retrieve(self):
        """Events can be added and retrieved."""
        dlq = DeadLetterQueue(max_size=10)
        event = Event(event_type=EventType.ORDER_SENT, sequence_num=1, order_id="test")
        dlq.add(event, "test error")

        entries = dlq.get_all()
        assert len(entries) == 1
        assert entries[0].event == event
        assert entries[0].error == "test error"

    def test_max_size_eviction(self):
        """Oldest entries are evicted when over capacity."""
        dlq = DeadLetterQueue(max_size=2)
        for i in range(5):
            dlq.add(Event(event_type=EventType.SHUTDOWN, sequence_num=i), f"error_{i}")

        assert dlq.size == 2
        entries = dlq.get_all()
        assert entries[0].error == "error_3"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_order_lifecycle(self, state_manager, validator, event_queue):
        """Test complete order lifecycle through events."""
        consumer = create_consumer(state_manager, validator, event_queue)

        # Create attempt and order
        attempt = state_manager.create_attempt(strategy_path=1, target_qty=10)
        order = state_manager.create_order(
            venue="kalshi",
            market_id="TEST",
            side="buy",
            price=0.50,
            quantity=10,
            attempt_id=attempt.attempt_id,
        )

        # Queue events for full lifecycle
        events = [
            Event(event_type=EventType.ATTEMPT_EXECUTE, sequence_num=1, attempt_id=attempt.attempt_id),
            Event(event_type=EventType.ORDER_SENT, sequence_num=2, venue="kalshi", order_id=order.order_id),
            Event(event_type=EventType.ORDER_ACKED, sequence_num=3, venue="kalshi", order_id=order.order_id, venue_order_id="K123"),
            Event(event_type=EventType.ORDER_FILL, sequence_num=4, venue="kalshi", order_id=order.order_id, fill_qty=10, fill_price=0.50),
            Event(event_type=EventType.ATTEMPT_RECONCILE, sequence_num=5, attempt_id=attempt.attempt_id),
            Event(event_type=EventType.ATTEMPT_COMPLETE, sequence_num=6, attempt_id=attempt.attempt_id),
            Event(event_type=EventType.SHUTDOWN, sequence_num=99),
        ]
        for e in events:
            event_queue.enqueue(e)

        await consumer.run()

        # Verify final states
        final_order = state_manager.get_order(order.order_id)
        assert final_order.state == OrderState.FILLED
        assert final_order.filled_qty == 10

        final_attempt = state_manager.get_attempt(attempt.attempt_id)
        # Note: ATTEMPT_COMPLETE leads to DONE_BALANCED or DONE_IMBALANCED based on fills
        # With only Kalshi filled (poly=0), it should be DONE_IMBALANCED
        assert final_attempt.state in (AttemptState.DONE_BALANCED, AttemptState.DONE_IMBALANCED)

        # Verify inventory updated
        inv = state_manager.get_inventory()
        assert inv.kalshi_position == 10

    @pytest.mark.asyncio
    async def test_balanced_arb_attempt(self, state_manager, validator, event_queue):
        """Test balanced arb with fills on both venues."""
        consumer = create_consumer(state_manager, validator, event_queue)

        attempt = state_manager.create_attempt(strategy_path=1, target_qty=10)
        kalshi_order = state_manager.create_order("kalshi", "TEST-K", "buy", 0.50, 10, attempt.attempt_id)
        poly_order = state_manager.create_order("poly", "TEST-P", "buy", 0.48, 10, attempt.attempt_id)

        # Update attempt with order IDs
        state_manager.update_attempt_state(
            attempt.attempt_id,
            AttemptState.EXECUTING,
            kalshi_order_id=kalshi_order.order_id,
            poly_order_id=poly_order.order_id,
        )

        # Queue both order lifecycles
        seq = 1
        for order, venue in [(kalshi_order, "kalshi"), (poly_order, "poly")]:
            event_queue.enqueue(Event(event_type=EventType.ORDER_SENT, sequence_num=seq, venue=venue, order_id=order.order_id))
            seq += 1
            event_queue.enqueue(Event(event_type=EventType.ORDER_ACKED, sequence_num=seq, venue=venue, order_id=order.order_id, venue_order_id=f"V{seq}"))
            seq += 1
            event_queue.enqueue(Event(event_type=EventType.ORDER_FILL, sequence_num=seq, venue=venue, order_id=order.order_id, fill_qty=10, fill_price=order.price))
            seq += 1

        event_queue.enqueue(Event(event_type=EventType.ATTEMPT_RECONCILE, sequence_num=seq, attempt_id=attempt.attempt_id))
        event_queue.enqueue(Event(event_type=EventType.ATTEMPT_COMPLETE, sequence_num=seq+1, attempt_id=attempt.attempt_id))
        event_queue.enqueue(Event(event_type=EventType.SHUTDOWN, sequence_num=99))

        await consumer.run()

        # Both orders should be filled
        assert state_manager.get_order(kalshi_order.order_id).state == OrderState.FILLED
        assert state_manager.get_order(poly_order.order_id).state == OrderState.FILLED

        # Attempt should be balanced
        final_attempt = state_manager.get_attempt(attempt.attempt_id)
        assert final_attempt.kalshi_filled == 10
        assert final_attempt.poly_filled == 10
        assert final_attempt.is_balanced
        assert final_attempt.state == AttemptState.DONE_BALANCED

    @pytest.mark.asyncio
    async def test_consumer_metrics(self, consumer, event_queue):
        """Consumer tracks metrics correctly."""
        for i in range(10):
            event_queue.enqueue(Event(event_type=EventType.INVENTORY_UPDATE, sequence_num=i))
        event_queue.enqueue(Event(event_type=EventType.SHUTDOWN, sequence_num=99))

        await consumer.run()

        assert consumer.metrics.events_processed >= 10
        assert consumer.metrics.uptime_ms > 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

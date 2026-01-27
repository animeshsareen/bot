"""Optional state management system for cross-venue arbitrage.

Enable by setting ENABLE_STATE_MANAGER=true in environment.
When disabled, the bot runs in lean mode without persistence or event-driven state.

Usage:
    from .state_manager import StateManager, LockManager  # always available
    from .state_manager import EventConsumer, CommandExecutor  # only when enabled
"""

import os

ENABLE_STATE_MANAGER = os.environ.get("ENABLE_STATE_MANAGER", "false").lower() == "true"

# Always export lightweight types (no heavy deps)
from .types import (
    OrderState,
    AttemptState,
    EventType,
    Order,
    Attempt,
    Inventory,
    Event,
    Snapshot,
    Fill,
    ORDER_TRANSITIONS,
    ATTEMPT_TRANSITIONS,
    TERMINAL_ORDER_STATES,
    is_valid_order_transition,
    is_valid_attempt_transition,
)
from .commands import SubmitOrderCommand, CancelOrderCommand, FlattenCommand, Command
from .locks import LockManager, LockTimeoutError

# Conditionally export heavy modules
if ENABLE_STATE_MANAGER:
    from .state import StateManager, StateError, InvalidTransitionError, OutOfOrderEventError
    from .event_validator import EventValidator, ValidationResult, BoundedEventQueue
    from .state_consumer import EventConsumer, EventHandlers, DeadLetterQueue
    from .command_executor import CommandQueue, CommandExecutor
    from .reconciliation_loop import ReconciliationLoop

"""
Simple event bus for inter-component communication.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from core.logger import get_logger

log = get_logger("event_bus")


class EventBus:
    """
    Simple publish-subscribe event bus for async event handling.

    Usage:
        bus = EventBus()

        async def on_price_update(data):
            print(f"Price: {data['price']}")

        bus.subscribe("price_update", on_price_update)
        await bus.publish("price_update", {"price": 100.0})
    """

    def __init__(self) -> None:
        """Initialize the event bus with empty subscriber lists."""
        self._subscribers: dict[str, list[Callable[..., Coroutine[Any, Any, None]]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def subscribe(
        self,
        event_type: str,
        handler: Callable[..., Coroutine[Any, Any, None]]
    ) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Event name to subscribe to
            handler: Async function to call when event is published
        """
        self._subscribers[event_type].append(handler)
        log.debug(f"Subscribed handler to '{event_type}' event")

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[..., Coroutine[Any, Any, None]]
    ) -> None:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: Event name to unsubscribe from
            handler: Handler function to remove
        """
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            log.debug(f"Unsubscribed handler from '{event_type}' event")

    async def publish(self, event_type: str, data: Any = None) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Event name to publish
            data: Event data to pass to handlers
        """
        handlers = self._subscribers.get(event_type, [])

        if not handlers:
            log.debug(f"No subscribers for '{event_type}' event")
            return

        log.debug(f"Publishing '{event_type}' to {len(handlers)} subscriber(s)")

        # Run all handlers concurrently
        tasks = [handler(data) for handler in handlers]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            log.error(f"Error publishing '{event_type}' event: {e}")

    def clear(self, event_type: str | None = None) -> None:
        """
        Clear all subscribers for an event type or all events.

        Args:
            event_type: Event name to clear, or None to clear all
        """
        if event_type:
            self._subscribers[event_type] = []
            log.debug(f"Cleared subscribers for '{event_type}' event")
        else:
            self._subscribers.clear()
            log.debug("Cleared all event subscribers")


# Global event bus instance
event_bus = EventBus()

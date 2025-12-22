"""
Telegram notifications for AI Trading System V3.
Sends alerts, trade notifications, and status updates.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from core.logger import get_logger

log = get_logger("telegram")


class TelegramNotifier:
    """
    Telegram notification sender with rate limiting and retry logic.

    Features:
    - Rate limiting (1 message per second)
    - Retry with exponential backoff
    - Message deduplication (same message not sent within cooldown)
    - Different message types (alert, trade, error)
    - Emoji formatting
    """

    # Rate limit: minimum seconds between messages
    MIN_INTERVAL = 1.0

    # Deduplication: minimum seconds before resending same message type
    DEDUP_COOLDOWN = 3600  # 1 hour

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        """
        Initialize TelegramNotifier.

        Args:
            bot_token: Telegram bot token (or reads from TELEGRAM_BOT_TOKEN env)
            chat_id: Chat ID to send messages to (or reads from TELEGRAM_CHAT_ID env)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        self.last_send_time = 0.0
        self.enabled = bool(self.bot_token and self.chat_id)

        # Deduplication: track last time each message key was sent
        self._sent_messages: dict[str, float] = {}

        if self.enabled:
            log.info(f"[TELEGRAM] Initialized: chat_id={self.chat_id}")
        else:
            log.warning("[TELEGRAM] Disabled (missing bot_token or chat_id)")

    def _should_send(self, message_key: str) -> bool:
        """
        Check if a message should be sent based on deduplication.

        Args:
            message_key: Unique key for this type of message

        Returns:
            True if message should be sent
        """
        now = time.time()
        last_sent = self._sent_messages.get(message_key, 0)

        if now - last_sent < self.DEDUP_COOLDOWN:
            log.debug(f"[TELEGRAM] Skipping duplicate message: {message_key}")
            return False

        return True

    def _mark_sent(self, message_key: str) -> None:
        """Mark a message as sent for deduplication tracking."""
        self._sent_messages[message_key] = time.time()

        # Clean up old entries (older than 2 hours)
        cutoff = time.time() - 7200
        self._sent_messages = {
            k: v for k, v in self._sent_messages.items() if v > cutoff
        }

    def _rate_limit(self) -> None:
        """Apply rate limiting between messages."""
        elapsed = time.time() - self.last_send_time
        if elapsed < self.MIN_INTERVAL:
            time.sleep(self.MIN_INTERVAL - elapsed)

    def _send_message(
        self,
        text: str,
        parse_mode: str | None = None,
        disable_preview: bool = True,
        max_retries: int = 3,
    ) -> bool:
        """
        Send a message to Telegram.

        Args:
            text: Message text
            parse_mode: "Markdown" or "HTML" or None
            disable_preview: Disable link previews
            max_retries: Maximum retry attempts

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            log.debug("[TELEGRAM] Skipping send (disabled)")
            return False

        self._rate_limit()

        url = f"{self.base_url}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": disable_preview,
        }

        if parse_mode:
            payload["parse_mode"] = parse_mode

        last_error = None

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=10.0) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    result = response.json()

                if result.get("ok"):
                    self.last_send_time = time.time()
                    log.debug(f"[TELEGRAM] Message sent: {text[:50]}...")
                    return True
                else:
                    error_desc = result.get("description", "Unknown error")
                    log.error(f"[TELEGRAM] API error: {error_desc}")
                    return False

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:  # Rate limited
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    log.warning(f"[TELEGRAM] Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                elif e.response.status_code in [400, 403]:
                    log.error(f"[TELEGRAM] Send failed: {e.response.status_code}")
                    return False

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    log.warning(f"[TELEGRAM] Send attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)

        log.error(f"[TELEGRAM] Failed after {max_retries} attempts: {last_error}")
        return False

    def send_alert(
        self,
        message: str,
        emoji: str = "⚠️",
        symbol: str = "",
    ) -> bool:
        """
        Send an alert message.

        Args:
            message: Alert message
            emoji: Emoji prefix
            symbol: Trading symbol (optional)

        Returns:
            True if sent successfully
        """
        symbol_text = f" [{symbol}]" if symbol else ""
        text = f"{emoji} ALERT{symbol_text}\n\n{message}"
        return self._send_message(text)

    def send_critical_alert(self, message: str, symbol: str = "") -> bool:
        """
        Send a critical alert with high priority.

        Args:
            message: Critical message
            symbol: Trading symbol

        Returns:
            True if sent successfully
        """
        symbol_text = f" [{symbol}]" if symbol else ""
        text = f"🚨🚨🚨 CRITICAL{symbol_text}\n\n{message}\n\n⚠️ IMMEDIATE ACTION REQUIRED!"
        return self._send_message(text)

    def send_trade_opened(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        leverage: int = 1,
    ) -> bool:
        """
        Send trade opened notification.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            quantity: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            leverage: Leverage used

        Returns:
            True if sent successfully
        """
        emoji = "📈" if side.lower() == "long" else "📉"
        side_text = side.upper()

        lines = [
            f"{emoji} TRADE OPENED",
            "",
            f"Symbol: {symbol}",
            f"Side: {side_text}",
            f"Quantity: {quantity}",
            f"Entry: ${entry_price:.4f}",
            f"Leverage: {leverage}x",
        ]

        if stop_loss:
            lines.append(f"Stop Loss: ${stop_loss:.4f}")
        if take_profit:
            lines.append(f"Take Profit: ${take_profit:.4f}")

        text = "\n".join(lines)
        return self._send_message(text)

    def send_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        duration_minutes: float,
        reason: str = "Manual",
    ) -> bool:
        """
        Send trade closed notification.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss in USDT
            pnl_pct: Profit/loss percentage
            duration_minutes: Trade duration in minutes
            reason: Close reason (TP, SL, Manual, etc.)

        Returns:
            True if sent successfully
        """
        emoji = "✅" if pnl >= 0 else "❌"
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"

        lines = [
            f"{emoji} TRADE CLOSED",
            "",
            f"Symbol: {symbol}",
            f"Side: {side.upper()}",
            f"Entry: ${entry_price:.4f}",
            f"Exit: ${exit_price:.4f}",
            f"PnL: {pnl_emoji} ${pnl:+.2f} ({pnl_pct:+.2f}%)",
            f"Duration: {duration_minutes:.1f} min",
            f"Reason: {reason}",
        ]

        text = "\n".join(lines)
        return self._send_message(text)

    def send_error(self, error_message: str, context: str = "") -> bool:
        """
        Send error notification.

        Args:
            error_message: Error message
            context: Context where error occurred

        Returns:
            True if sent successfully
        """
        context_text = f"\nContext: {context}" if context else ""
        text = f"❌ ERROR{context_text}\n\n{error_message}"
        return self._send_message(text)

    def send_status(
        self,
        balance: float,
        open_positions: int,
        daily_pnl: float,
        daily_pnl_pct: float,
    ) -> bool:
        """
        Send status update.

        Args:
            balance: Current balance
            open_positions: Number of open positions
            daily_pnl: Daily PnL in USDT
            daily_pnl_pct: Daily PnL percentage

        Returns:
            True if sent successfully
        """
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        now = datetime.now(timezone.utc).strftime("%H:%M UTC")

        lines = [
            f"📊 STATUS UPDATE ({now})",
            "",
            f"Balance: ${balance:.2f}",
            f"Open Positions: {open_positions}",
            f"Daily PnL: {pnl_emoji} ${daily_pnl:+.2f} ({daily_pnl_pct:+.2f}%)",
        ]

        text = "\n".join(lines)
        return self._send_message(text)

    def send_test_message(self) -> bool:
        """
        Send a test message to verify configuration.

        Returns:
            True if sent successfully
        """
        text = (
            "✅ AI Trading System V3\n\n"
            "Telegram notifications configured successfully!\n"
            f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        return self._send_message(text)

    async def send_message(self, text: str, dedup_key: str | None = None) -> bool:
        """
        Send a generic message (async wrapper).

        Args:
            text: Message text
            dedup_key: Optional key for deduplication. If provided, message won't
                      be sent again within DEDUP_COOLDOWN if same key was used.

        Returns:
            True if sent successfully
        """
        # Check deduplication
        if dedup_key and not self._should_send(dedup_key):
            return False

        # Run synchronous _send_message in executor to not block
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._send_message, text)

        # Mark as sent for deduplication
        if result and dedup_key:
            self._mark_sent(dedup_key)

        return result

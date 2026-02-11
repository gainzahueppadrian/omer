"""
Message Bus for Multi-Agent Communication

Provides a thread-safe pub/sub messaging system for agent coordination.
Agents publish messages to topics and subscribe to topics they care about.
The orchestrator uses this to coordinate workflows between agents.
"""

import threading
import queue
import time
import logging
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    CRITICAL = 0   # Risk breaches, emergency stops
    HIGH = 1       # Trade signals, execution requests
    NORMAL = 2     # Scan results, routine updates
    LOW = 3        # Heartbeats, diagnostics


@dataclass
class AgentMessage:
    """Message passed between agents."""
    topic: str
    sender: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


class MessageBus:
    """
    Central message bus for agent-to-agent communication.

    Features:
    - Priority-based message delivery
    - Topic-based pub/sub
    - Request-reply pattern support
    - Message history for audit trail
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queues: Dict[str, queue.PriorityQueue] = {}
        self._lock = threading.Lock()
        self._history: List[AgentMessage] = []
        self._max_history = max_history
        self._running = True

        # Reply channels for request-reply pattern
        self._reply_channels: Dict[str, queue.Queue] = {}

    def subscribe(self, topic: str, callback: Callable[[AgentMessage], None]):
        """Subscribe a callback to a topic."""
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(callback)
            logger.debug(f"Subscription added: {topic}")

    def create_agent_queue(self, agent_name: str) -> queue.PriorityQueue:
        """Create a dedicated message queue for an agent."""
        q = queue.PriorityQueue()
        self._queues[agent_name] = q
        return q

    def publish(self, message: AgentMessage):
        """Publish a message to all subscribers of the topic."""
        with self._lock:
            # Store in history
            self._history.append(message)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Deliver to topic subscribers
        subscribers = self._subscribers.get(message.topic, [])
        for callback in subscribers:
            try:
                callback(message)
            except Exception as e:
                logger.error(
                    f"Error delivering message to subscriber "
                    f"(topic={message.topic}): {e}"
                )

        # Deliver to agent queues if topic matches agent name
        for agent_name, q in self._queues.items():
            if message.topic == agent_name or message.topic == "broadcast":
                q.put((message.priority.value, time.time(), message))

        # Handle reply channels
        if message.correlation_id and message.correlation_id in self._reply_channels:
            self._reply_channels[message.correlation_id].put(message)

        logger.debug(
            f"Message published: {message.sender} -> {message.topic} "
            f"(priority={message.priority.name})"
        )

    def request_reply(self, message: AgentMessage,
                       timeout: float = 30.0) -> Optional[AgentMessage]:
        """
        Send a message and wait for a reply (synchronous request-reply pattern).
        """
        import uuid
        correlation_id = str(uuid.uuid4())
        message.correlation_id = correlation_id

        reply_queue = queue.Queue()
        self._reply_channels[correlation_id] = reply_queue

        self.publish(message)

        try:
            reply = reply_queue.get(timeout=timeout)
            return reply
        except queue.Empty:
            logger.warning(
                f"Request-reply timeout for correlation_id={correlation_id}"
            )
            return None
        finally:
            self._reply_channels.pop(correlation_id, None)

    def get_history(self, topic: Optional[str] = None,
                    limit: int = 50) -> List[AgentMessage]:
        """Get message history, optionally filtered by topic."""
        with self._lock:
            if topic:
                filtered = [m for m in self._history if m.topic == topic]
            else:
                filtered = list(self._history)
            return filtered[-limit:]

    def shutdown(self):
        """Shut down the message bus."""
        self._running = False
        logger.info("Message bus shutting down")

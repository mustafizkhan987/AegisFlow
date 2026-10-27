"""
AegisFlow Observability Package
===============================
Exposes OpenTelemetry tracer and the WebSocket hub.
"""

from aegisflow.observability.tracer import tracer, trace_span
from aegisflow.observability.websocket_hub import ws_hub, WebSocketHub

__all__ = ["tracer", "trace_span", "ws_hub", "WebSocketHub"]

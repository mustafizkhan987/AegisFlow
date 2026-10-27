"""
AegisFlow WebSocket Observability Hub
=====================================
Streams live execution graphs, state machine transitions, tool calls,
sandbox terminal logs, and cost telemetry to the React dashboard.
"""

import asyncio
import json
import time
from typing import Set, Dict, Any, List
from fastapi import WebSocket


class WebSocketHub:
    """Manages active WebSockets and broadcasts live telemetry events."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.recent_events: List[Dict[str, Any]] = []
        self.stats = {
            "total_incidents": 0,
            "transient_flakes_filtered": 0,
            "dependency_conflicts_routed": 0,
            "code_regressions_repaired": 0,
            "total_tokens_saved": 0,
            "total_cost_saved_usd": 0.0,
            "avg_triage_latency_ms": 12.4,
            "pass_at_1_rate": 88.5
        }

    async def connect(self, websocket: WebSocket):
        """Accepts and registers a new client WebSocket."""
        await websocket.accept()
        self.active_connections.add(websocket)

        # Send initial snapshot of stats and recent events
        await websocket.send_json({
            "event_type": "INITIAL_SNAPSHOT",
            "timestamp": time.time(),
            "payload": {
                "stats": self.stats,
                "recent_events": self.recent_events[-20:]
            }
        })

    def disconnect(self, websocket: WebSocket):
        """Removes a disconnected client."""
        self.active_connections.discard(websocket)

    async def broadcast(self, event_type: str, payload: Dict[str, Any]):
        """Broadcasts an event message to all connected dashboard clients."""
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "payload": payload
        }
        self.recent_events.append(event)
        if len(self.recent_events) > 100:
            self.recent_events.pop(0)

        # Update running cumulative stats
        if event_type == "TRIAGE_COMPLETED":
            self.stats["total_incidents"] += 1
            cat = payload.get("category", "")
            if cat == "TRANSIENT_INFRA_FLAKE":
                self.stats["transient_flakes_filtered"] += 1
            elif cat == "DEPENDENCY_CONFLICT":
                self.stats["dependency_conflicts_routed"] += 1
            elif cat == "CODE_REGRESSION":
                pass

            savings = payload.get("estimated_token_savings", 0)
            self.stats["total_tokens_saved"] += savings
            self.stats["total_cost_saved_usd"] = round(
                self.stats["total_tokens_saved"] * (0.003 / 1000.0), 3
            )
            lat = payload.get("latency_ms", 12.0)
            self.stats["avg_triage_latency_ms"] = round(
                (self.stats["avg_triage_latency_ms"] * 0.9) + (lat * 0.1), 2
            )

        elif event_type == "REMEDIATION_FINISHED" and payload.get("status") == "RESOLVED":
            self.stats["code_regressions_repaired"] += 1

        dead_connections = set()
        for connection in list(self.active_connections):
            try:
                await connection.send_json(event)
            except Exception:
                dead_connections.add(connection)

        for dead in dead_connections:
            self.active_connections.discard(dead)


# Global singleton hub
ws_hub = WebSocketHub()

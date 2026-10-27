"""
AegisFlow OpenTelemetry Tracing & Metrics
=========================================
Instruments every agent step, tool call, token count, and sandbox execution span
with OpenTelemetry Python.
"""

from typing import Optional, Dict, Any
from contextlib import contextmanager
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

from aegisflow.config import settings

# Initialize OTel Tracer Provider
resource = Resource.create({"service.name": settings.app_name, "service.version": settings.app_version})
provider = TracerProvider(resource=resource)

# In debug mode, print traces to console
if settings.debug:
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("aegisflow.tracer", settings.app_version)


@contextmanager
def trace_span(span_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for tracing execution spans with custom attributes."""
    if not settings.enable_opentelemetry:
        yield None
        return

    with tracer.start_as_current_span(span_name) as span:
        if attributes:
            for k, v in attributes.items():
                if isinstance(v, (str, bool, int, float)):
                    span.set_attribute(k, v)
                else:
                    span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise

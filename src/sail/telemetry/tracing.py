from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
    _append_trace_path,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace.propagation import set_span_in_context


def trace_as_current_with_action(span_name):
    def trace_span(func):
        def wrapper(*args, **kwargs):
            base_class = args[0]

            if base_class.tracer is not None:
                verbosity = base_class.verbosity
                with base_class.tracer.trace_as_current(
                    span_name=f"Epoch-{verbosity.current_epoch_n}-{span_name}",
                    verbose=verbosity.get(),
                ):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return trace_span


def trace_with_action(span_name):
    def trace_span(func):
        def wrapper(*args, **kwargs):
            base_class = args[0]

            if base_class.tracer is not None:
                verbosity = base_class.verbosity
                with base_class.tracer.trace(
                    span_name=f"Epoch-{verbosity.current_epoch_n}-{span_name}",
                    verbose=verbosity.get(),
                ):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return trace_span


def trace_step(span_name):
    def trace_span(func):
        def wrapper(*args, **kwargs):
            base_class = args[0]

            if base_class.tracer is not None:
                with base_class.tracer.trace(span_name=span_name):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return trace_span


class TracingClient:
    def __init__(
        self,
        service_name: str,
        oltp_endpoint: str = None,
        span_exporter: SpanExporter = None,
    ) -> None:
        self.tracer_name = service_name
        self.span_exporter = span_exporter

        provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: service_name})
        )

        if oltp_endpoint is not None:
            span_exporter = OTLPSpanExporter(endpoint=_append_trace_path(oltp_endpoint))
        elif span_exporter is None:
            span_exporter = OTLPSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(provider)

        self._tracer = trace.get_tracer(service_name)

    def trace_as_current(
        self, span_name, kind=trace.SpanKind.INTERNAL, context=None, verbose=1, **kwargs
    ):
        if verbose == 1:
            if hasattr(self, "_parent_span"):
                context = set_span_in_context(self._parent_span)
            return self._tracer.start_as_current_span(
                span_name, context=context, kind=kind
            )
        else:
            return DummySpan()

    def trace(
        self, span_name, kind=trace.SpanKind.INTERNAL, context=None, verbose=1, **kwargs
    ):
        if verbose == 1:
            if hasattr(self, "_parent_span"):
                context = set_span_in_context(self._parent_span)
            return self._tracer.start_span(span_name, context=context, kind=kind)
        else:
            return DummySpan()

    def set_attribute(self, name, value):
        trace.get_current_span().set_attribute(name, value)


class DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self

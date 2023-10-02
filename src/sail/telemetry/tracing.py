from opentelemetry import trace as oltp_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
    _append_trace_path,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace.propagation import set_span_in_context

from sail.utils.url_utils import validate_address


def trace_with_action(span_name, current_span=False):
    def trace_span(func):
        def wrapper(*args, **kwargs):
            base_class = args[0]

            if base_class.tracer is not None:
                verbosity = base_class.verbosity
                if current_span:
                    with base_class.tracer.trace_as_current_span(
                        span_name=f"Epoch-{verbosity.current_epoch_n}-{span_name}",
                        verbose=verbosity.get(),
                    ):
                        return func(*args, **kwargs)
                else:
                    with base_class.tracer.trace(
                        span_name=f"Epoch-{verbosity.current_epoch_n}-{span_name}",
                        verbose=verbosity.get(),
                    ):
                        return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return trace_span


def trace(span_name, current_span=False):
    def trace_span(func):
        def wrapper(*args, **kwargs):
            base_class = args[0]

            if base_class.tracer is not None:
                if current_span:
                    with base_class.tracer.trace_as_current_span(span_name=span_name):
                        return func(*args, **kwargs)
                else:
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
        otlp_endpoint: str = None,
        span_exporter: SpanExporter = None,
    ) -> None:
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.span_exporter = span_exporter

        # verify if the tracer enpoint is valid
        try:
            validate_address(otlp_endpoint)
        except Exception as e:
            raise Exception(f"Tracing Error: {str(e)}")

        provider = TracerProvider(
            resource=Resource.create({SERVICE_NAME: service_name})
        )

        if otlp_endpoint is not None:
            span_exporter = OTLPSpanExporter(endpoint=_append_trace_path(otlp_endpoint))
        elif span_exporter is None:
            span_exporter = OTLPSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(span_exporter))
        oltp_trace.set_tracer_provider(provider)

        self._tracer = oltp_trace.get_tracer(service_name)

    def trace_as_current_span(
        self,
        span_name,
        kind=oltp_trace.SpanKind.INTERNAL,
        context=None,
        verbose=1,
    ):
        if verbose == 1:
            return self._tracer.start_as_current_span(
                span_name, context=context, kind=kind
            )
        else:
            return DummySpan()

    def trace(
        self,
        span_name,
        kind=oltp_trace.SpanKind.INTERNAL,
        context=None,
        verbose=1,
    ):
        if verbose == 1:
            return self._tracer.start_span(span_name, context=context, kind=kind)
        else:
            return DummySpan()

    def set_attribute(self, name, value):
        oltp_trace.get_current_span().set_attribute(name, value)


class DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self

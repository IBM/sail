import os


class TracingClient:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port

    def trace(self, *args, **kwargs):
        return DummySpan()


class DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self

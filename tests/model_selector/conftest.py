import pytest
import ray


@pytest.fixture()
def ray_setup():
    print("\nExecuting ray.init()")
    ray.init()
    yield "Ray Shutdown..."
    ray.shutdown()

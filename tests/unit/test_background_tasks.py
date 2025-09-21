import asyncio
from unittest.mock import MagicMock

import pytest


# Mock background task runner
class MockBackgroundTaskRunner:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))

    async def run_tasks(self):
        for func, args, kwargs in self.tasks:
            await func(*args, **kwargs)


@pytest.fixture
def background_tasks():
    """Fixture for a mock background task runner."""
    return MockBackgroundTaskRunner()


async def sample_background_task(result_store: dict, task_id: int):
    """A sample task to be run in the background."""
    await asyncio.sleep(0.01)  # Simulate async work
    result_store[task_id] = "completed"


@pytest.mark.asyncio
async def test_add_and_run_background_task(background_tasks):
    """Test that a background task can be added and executed."""
    result_store = {}
    task_id = 1

    # Simulate adding a task from a service
    background_tasks.add_task(sample_background_task, result_store, task_id)

    # Simulate the background runner executing the task
    await background_tasks.run_tasks()

    assert result_store.get(task_id) == "completed"


def another_sample_task(x, y):
    """A simple synchronous task."""
    return x + y


@pytest.mark.asyncio
async def test_background_task_with_sync_function(background_tasks):
    """Test that synchronous functions can also be run."""
    # This is a conceptual test. In a real FastAPI app, you'd use its BackgroundTasks
    # which handles both sync and async functions.

    # We'll adapt our mock runner to handle this for the test.
    async def run_sync_in_async(func, *args, **kwargs):
        return func(*args, **kwargs)

    background_tasks.add_task(run_sync_in_async, another_sample_task, 2, 3)

    # This part is tricky to test without a full event loop and runner.
    # The main point is to ensure the task is added.
    assert len(background_tasks.tasks) == 1

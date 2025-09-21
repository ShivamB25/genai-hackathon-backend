import pytest

from src.ai_services.function_tools import (
    EnhancedToolRegistry,
    ToolCategory,
    tool_function,
)


# Sample functions to be used as tools
@tool_function(
    name="get_weather",
    description="Gets the weather for a location.",
    category=ToolCategory.WEATHER,
)
def get_weather(location: str):
    """Gets the weather for a location."""
    return f"The weather in {location} is sunny."


@tool_function(
    name="get_flight_info",
    description="Gets information about a flight.",
    category=ToolCategory.TRAVEL,
)
def get_flight_info(flight_number: str):
    """Gets information about a flight."""
    return {"flight": flight_number, "status": "on time"}


@pytest.fixture
def registry():
    """Fixture for an EnhancedToolRegistry."""
    return EnhancedToolRegistry()


@pytest.fixture
def executor(registry):
    """Fixture for a ToolExecutor."""
    return registry


def test_register_tool(registry):
    """Test that a function can be registered as a tool."""
    assert "get_weather" in registry.list_tools()


@pytest.mark.asyncio
async def test_execute_tool(executor):
    """Test that a tool can be executed correctly."""
    result = await executor.execute_tool("get_weather", location="Paris")
    assert "The weather in Paris is sunny." in str(result)


@pytest.mark.asyncio
async def test_execute_tool_with_json_output(executor):
    """Test a tool that returns a dictionary."""
    result = await executor.execute_tool("get_flight_info", flight_number="AA123")
    assert "AA123" in str(result)


@pytest.mark.asyncio
async def test_execute_nonexistent_tool(executor):
    """Test that executing a nonexistent tool raises an error."""
    with pytest.raises(Exception):
        await executor.execute_tool("nonexistent_tool")


def test_get_tool_schema(registry):
    """Test that the tool schema is generated correctly."""
    tool = registry.get_tool("get_weather")
    assert tool.metadata.name == "get_weather"

"""Function Tools Foundation for AI-Powered Trip Planner Backend.

This module provides a framework for external API integration tools, basic utility tools,
tool registration and discovery system, and async tool execution with error handling.
"""

import asyncio
import inspect
import json
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx

from src.ai_services.exceptions import FunctionCallError
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ToolCategory(str, Enum):
    """Categories of function tools."""

    UTILITY = "utility"
    DATE_TIME = "date_time"
    VALIDATION = "validation"
    EXTERNAL_API = "external_api"
    MAPS = "maps"
    TRIP_PLANNING = "trip_planning"
    ITINERARY = "itinerary"
    OPTIMIZATION = "optimization"
    WEATHER = "weather"
    EVENTS = "events"
    ANALYTICS = "analytics"
    TRAVEL = "travel"
    PLACES = "places"


class ToolExecutionMode(str, Enum):
    """Tool execution modes."""

    SYNC = "sync"
    ASYNC = "async"
    BACKGROUND = "background"


@dataclass
class ToolParameter:
    """Function tool parameter definition."""

    name: str
    type_hint: type
    description: str
    required: bool = True
    default: Any = None
    example: Any = None
    validation_pattern: Optional[str] = None


@dataclass
class ToolMetadata:
    """Metadata for function tools."""

    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: Optional[type] = None
    execution_mode: ToolExecutionMode = ToolExecutionMode.ASYNC
    timeout_seconds: int = 30
    requires_auth: bool = False
    rate_limit: Optional[int] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    # Enhanced analytics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    usage_patterns: List[Dict[str, Any]] = field(default_factory=list)
    chaining_compatible: bool = False
    batch_capable: bool = False
    cacheable: bool = False


class FunctionTool(ABC):
    """Abstract base class for function tools."""

    def __init__(self, metadata: ToolMetadata) -> None:
        """Initialize function tool.

        Args:
            metadata: Tool metadata
        """
        self.metadata = metadata
        self._call_count = 0
        self._last_called = None
        self._rate_limit_window = {}

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool function.

        Args:
            **kwargs: Tool parameters

        Returns:
            Any: Tool execution result

        Raises:
            FunctionCallError: If execution fails
        """

    def validate_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters.

        Args:
            kwargs: Input parameters

        Returns:
            Dict[str, Any]: Validated parameters

        Raises:
            FunctionCallError: If validation fails
        """
        validated = {}

        # Check required parameters
        required_params = [p.name for p in self.metadata.parameters if p.required]
        missing_params = [p for p in required_params if p not in kwargs]

        if missing_params:
            raise FunctionCallError(
                f"Missing required parameters: {', '.join(missing_params)}",
                function_name=self.metadata.name,
                function_args=kwargs,
            )

        # Validate and convert parameter types
        for param in self.metadata.parameters:
            value = kwargs.get(param.name, param.default)

            if value is None and not param.required:
                continue

            try:
                # Basic type validation
                if param.type_hint == str and not isinstance(value, str):
                    value = str(value)
                elif param.type_hint == int and not isinstance(value, int):
                    value = int(value)
                elif param.type_hint == float and not isinstance(value, int | float):
                    value = float(value)
                elif param.type_hint == bool and not isinstance(value, bool):
                    value = bool(value)

                validated[param.name] = value

            except (ValueError, TypeError) as e:
                raise FunctionCallError(
                    f"Parameter '{param.name}' type validation failed: {e}",
                    function_name=self.metadata.name,
                    function_args=kwargs,
                ) from e

        return validated

    def check_rate_limit(self) -> None:
        """Check if tool is within rate limits.

        Raises:
            FunctionCallError: If rate limit exceeded
        """
        if not self.metadata.rate_limit:
            return

        now = datetime.now(timezone.utc)
        window_start = now.replace(minute=0, second=0, microsecond=0)

        # Clean old entries
        self._rate_limit_window = {
            ts: count
            for ts, count in self._rate_limit_window.items()
            if ts >= window_start
        }

        # Count calls in current window
        current_calls = sum(self._rate_limit_window.values())

        if current_calls >= self.metadata.rate_limit:
            raise FunctionCallError(
                f"Rate limit exceeded for tool '{self.metadata.name}': "
                f"{current_calls}/{self.metadata.rate_limit} calls per hour",
                function_name=self.metadata.name,
            )

        # Record this call
        self._rate_limit_window[window_start] = (
            self._rate_limit_window.get(window_start, 0) + 1
        )

    def update_call_stats(self) -> None:
        """Update tool call statistics."""
        self._call_count += 1
        self._last_called = datetime.now(timezone.utc)

    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.

        Returns:
            Dict[str, Any]: Usage statistics
        """
        return {
            "name": self.metadata.name,
            "call_count": self._call_count,
            "last_called": self._last_called.isoformat() if self._last_called else None,
            "rate_limit": self.metadata.rate_limit,
            "current_window_calls": sum(self._rate_limit_window.values()),
        }


class UtilityTools:
    """Collection of basic utility tools."""

    @staticmethod
    def create_current_datetime_tool() -> FunctionTool:
        """Create current datetime tool."""

        class CurrentDateTimeTool(FunctionTool):
            async def execute(self, **kwargs) -> Dict[str, Any]:
                validated_params = self.validate_parameters(kwargs)

                timezone_str = validated_params.get("timezone", "UTC")
                format_str = validated_params.get("format", "%Y-%m-%d %H:%M:%S")

                try:
                    # Handle timezone
                    if timezone_str == "UTC":
                        dt = datetime.now(timezone.utc)
                    else:
                        # For simplicity, using UTC for now
                        # In production, you'd want proper timezone handling
                        dt = datetime.now(timezone.utc)

                    return {
                        "datetime": dt.strftime(format_str),
                        "iso_format": dt.isoformat(),
                        "timestamp": dt.timestamp(),
                        "timezone": timezone_str,
                        "date": dt.strftime("%Y-%m-%d"),
                        "time": dt.strftime("%H:%M:%S"),
                        "day_of_week": dt.strftime("%A"),
                        "month_name": dt.strftime("%B"),
                    }

                except Exception as e:
                    raise FunctionCallError(
                        f"Failed to get current datetime: {e}"
                    ) from e

        metadata = ToolMetadata(
            name="get_current_datetime",
            description="Get current date and time information",
            category=ToolCategory.DATE_TIME,
            parameters=[
                ToolParameter(
                    name="timezone",
                    type_hint=str,
                    description="Timezone (e.g., 'UTC', 'Asia/Kolkata')",
                    required=False,
                    default="UTC",
                    example="Asia/Kolkata",
                ),
                ToolParameter(
                    name="format",
                    type_hint=str,
                    description="Date format string",
                    required=False,
                    default="%Y-%m-%d %H:%M:%S",
                    example="%d/%m/%Y %I:%M %p",
                ),
            ],
            examples=[
                {"timezone": "UTC"},
                {"timezone": "Asia/Kolkata", "format": "%d/%m/%Y %I:%M %p"},
            ],
        )

        return CurrentDateTimeTool(metadata)

    @staticmethod
    def create_validate_email_tool() -> FunctionTool:
        """Create email validation tool."""

        class ValidateEmailTool(FunctionTool):
            async def execute(self, **kwargs) -> Dict[str, Any]:
                validated_params = self.validate_parameters(kwargs)
                email = validated_params["email"]

                # Simple email validation
                import re

                pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                is_valid = bool(re.match(pattern, email))

                return {
                    "email": email,
                    "is_valid": is_valid,
                    "domain": email.split("@")[1] if "@" in email else None,
                    "local_part": email.split("@")[0] if "@" in email else email,
                }

        metadata = ToolMetadata(
            name="validate_email",
            description="Validate email address format",
            category=ToolCategory.VALIDATION,
            parameters=[
                ToolParameter(
                    name="email",
                    type_hint=str,
                    description="Email address to validate",
                    required=True,
                    example="user@example.com",
                ),
            ],
            examples=[
                {"email": "user@example.com"},
                {"email": "invalid-email"},
            ],
        )

        return ValidateEmailTool(metadata)

    @staticmethod
    def create_currency_converter_tool() -> FunctionTool:
        """Create basic currency converter tool."""

        class CurrencyConverterTool(FunctionTool):
            async def execute(self, **kwargs) -> Dict[str, Any]:
                validated_params = self.validate_parameters(kwargs)

                amount = validated_params["amount"]
                from_currency = validated_params["from_currency"].upper()
                to_currency = validated_params["to_currency"].upper()

                # Mock exchange rates (in production, use real API)
                mock_rates = {
                    "USD": 1.0,
                    "EUR": 0.85,
                    "GBP": 0.73,
                    "INR": 83.12,
                    "JPY": 149.50,
                }

                if from_currency not in mock_rates or to_currency not in mock_rates:
                    raise FunctionCallError(
                        f"Unsupported currency: {from_currency} or {to_currency}"
                    )

                # Convert via USD
                usd_amount = amount / mock_rates[from_currency]
                converted_amount = usd_amount * mock_rates[to_currency]

                return {
                    "original_amount": amount,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "converted_amount": round(converted_amount, 2),
                    "exchange_rate": round(
                        mock_rates[to_currency] / mock_rates[from_currency], 4
                    ),
                    "note": "Mock exchange rates for demonstration",
                }

        metadata = ToolMetadata(
            name="convert_currency",
            description="Convert amount between currencies",
            category=ToolCategory.UTILITY,
            parameters=[
                ToolParameter(
                    name="amount",
                    type_hint=float,
                    description="Amount to convert",
                    required=True,
                    example=100.0,
                ),
                ToolParameter(
                    name="from_currency",
                    type_hint=str,
                    description="Source currency code",
                    required=True,
                    example="USD",
                ),
                ToolParameter(
                    name="to_currency",
                    type_hint=str,
                    description="Target currency code",
                    required=True,
                    example="INR",
                ),
            ],
            examples=[
                {"amount": 100, "from_currency": "USD", "to_currency": "INR"},
                {"amount": 50, "from_currency": "EUR", "to_currency": "GBP"},
            ],
        )

        return CurrencyConverterTool(metadata)


class ExternalAPITool(FunctionTool):
    """Base class for external API integration tools."""

    def __init__(
        self, metadata: ToolMetadata, base_url: str, api_key: Optional[str] = None
    ) -> None:
        """Initialize external API tool.

        Args:
            metadata: Tool metadata
            base_url: API base URL
            api_key: API key for authentication
        """
        super().__init__(metadata)
        self.base_url = base_url
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            httpx.AsyncClient: HTTP client instance
        """
        if not self._client:
            headers = {"User-Agent": f"{settings.app_name}/{settings.app_version}"}

            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(self.metadata.timeout_seconds),
            )

        return self._client

    async def make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request to external API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Request parameters

        Returns:
            Dict[str, Any]: API response

        Raises:
            FunctionCallError: If request fails
        """
        client = await self.get_client()

        try:
            response = await client.request(method, endpoint, **kwargs)
            response.raise_for_status()

            return response.json()

        except httpx.TimeoutException as e:
            raise FunctionCallError(
                f"API request timeout: {e}", function_name=self.metadata.name
            ) from e

        except httpx.HTTPStatusError as e:
            raise FunctionCallError(
                f"API request failed with status {e.response.status_code}: {e.response.text}",
                function_name=self.metadata.name,
            ) from e

        except Exception as e:
            raise FunctionCallError(
                f"API request failed: {e}", function_name=self.metadata.name
            ) from e

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class EnhancedFunctionTool(FunctionTool):
    """Enhanced function tool with analytics and chaining capabilities."""

    def __init__(self, metadata: ToolMetadata) -> None:
        super().__init__(metadata)
        self._execution_history: List[Dict[str, Any]] = []
        self._performance_stats = {
            "avg_execution_time": 0.0,
            "success_rate": 100.0,
            "total_executions": 0,
            "cache_hits": 0,
            "error_patterns": {},
        }
        self._result_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes default

    async def execute_with_analytics(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with enhanced analytics tracking."""
        execution_start = datetime.now(timezone.utc)
        execution_id = str(uuid4())[:8]

        try:
            # Check cache first
            cache_key = self._generate_cache_key(kwargs)
            cached_result = self._get_cached_result(cache_key)

            if cached_result:
                self._performance_stats["cache_hits"] += 1
                return {
                    "tool_name": self.metadata.name,
                    "result": cached_result,
                    "execution_time": 0.001,  # Cache hit
                    "cache_hit": True,
                    "execution_id": execution_id,
                    "timestamp": execution_start.isoformat(),
                }

            # Execute the tool
            result = await self.execute(**kwargs)

            execution_time = (
                datetime.now(timezone.utc) - execution_start
            ).total_seconds()

            # Cache result if cacheable
            if self.metadata.cacheable:
                self._cache_result(cache_key, result)

            # Record execution
            execution_record = {
                "execution_id": execution_id,
                "timestamp": execution_start.isoformat(),
                "execution_time": execution_time,
                "parameters": kwargs,
                "result_size": len(str(result)) if result else 0,
                "success": True,
            }

            self._execution_history.append(execution_record)
            self._update_performance_stats(execution_time, True)

            return {
                "tool_name": self.metadata.name,
                "result": result,
                "execution_time": execution_time,
                "cache_hit": False,
                "execution_id": execution_id,
                "timestamp": execution_start.isoformat(),
            }

        except Exception as e:
            execution_time = (
                datetime.now(timezone.utc) - execution_start
            ).total_seconds()

            error_type = type(e).__name__
            self._performance_stats["error_patterns"][error_type] = (
                self._performance_stats["error_patterns"].get(error_type, 0) + 1
            )

            execution_record = {
                "execution_id": execution_id,
                "timestamp": execution_start.isoformat(),
                "execution_time": execution_time,
                "parameters": kwargs,
                "error": str(e),
                "error_type": error_type,
                "success": False,
            }

            self._execution_history.append(execution_record)
            self._update_performance_stats(execution_time, False)

            raise FunctionCallError(
                f"Enhanced tool execution failed: {e}",
                function_name=self.metadata.name,
                function_args=kwargs,
            ) from e

    def _generate_cache_key(self, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for parameters."""
        import hashlib

        # Sort parameters for consistent hashing
        sorted_params = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.blake2s(sorted_params.encode(), digest_size=16).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Any:
        """Get cached result if not expired."""
        if cache_key in self._result_cache:
            result, cached_at = self._result_cache[cache_key]
            if datetime.now(timezone.utc) - cached_at < timedelta(
                seconds=self._cache_ttl
            ):
                return result
            else:
                # Remove expired entry
                del self._result_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache execution result."""
        self._result_cache[cache_key] = (result, datetime.now(timezone.utc))

        # Clean up old cache entries (keep last 100)
        if len(self._result_cache) > 100:
            oldest_key = min(
                self._result_cache.keys(), key=lambda k: self._result_cache[k][1]
            )
            del self._result_cache[oldest_key]

    def _update_performance_stats(self, execution_time: float, _success: bool) -> None:
        """Update performance statistics."""
        self._performance_stats["total_executions"] += 1

        # Update average execution time
        current_avg = self._performance_stats["avg_execution_time"]
        total_execs = self._performance_stats["total_executions"]
        new_avg = ((current_avg * (total_execs - 1)) + execution_time) / total_execs
        self._performance_stats["avg_execution_time"] = new_avg

        # Update success rate
        successful_executions = sum(
            1 for record in self._execution_history if record.get("success", False)
        )
        self._performance_stats["success_rate"] = (
            successful_executions / total_execs
        ) * 100

    def get_analytics(self) -> Dict[str, Any]:
        """Get detailed analytics for this tool."""
        recent_history = self._execution_history[-50:]  # Last 50 executions

        if not recent_history:
            return {"message": "No execution history available"}

        execution_times = [
            r["execution_time"] for r in recent_history if "execution_time" in r
        ]

        analytics = {
            **self._performance_stats,
            "recent_executions": len(recent_history),
            "execution_time_stats": {
                "min": min(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0,
                "median": statistics.median(execution_times) if execution_times else 0,
                "std_dev": (
                    statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                ),
            },
            "cache_stats": {
                "cache_size": len(self._result_cache),
                "cache_hit_rate": (
                    self._performance_stats["cache_hits"]
                    / max(self._performance_stats["total_executions"], 1)
                )
                * 100,
            },
            "recent_errors": [
                r for r in recent_history[-10:] if not r.get("success", True)
            ],
        }

        return analytics


class TripPlanningTools:
    """Collection of trip planning specific tools."""

    @staticmethod
    def create_budget_calculator_tool() -> EnhancedFunctionTool:
        """Create budget calculation tool for trip planning."""

        class BudgetCalculatorTool(EnhancedFunctionTool):
            async def execute(self, **kwargs) -> Dict[str, Any]:
                validated_params = self.validate_parameters(kwargs)

                destination = validated_params["destination"]
                duration_days = validated_params["duration_days"]
                traveler_count = validated_params.get("traveler_count", 1)
                validated_params.get("trip_type", "leisure")
                accommodation_level = validated_params.get(
                    "accommodation_level", "mid-range"
                )

                # Base daily costs (mock data - in production, use real pricing APIs)
                base_costs = {
                    "budget": {
                        "accommodation": 25,
                        "food": 15,
                        "activities": 10,
                        "transport": 8,
                    },
                    "mid-range": {
                        "accommodation": 75,
                        "food": 35,
                        "activities": 25,
                        "transport": 20,
                    },
                    "luxury": {
                        "accommodation": 200,
                        "food": 80,
                        "activities": 60,
                        "transport": 50,
                    },
                }

                # Destination multipliers (mock data)
                destination_multipliers = {
                    "mumbai": 1.2,
                    "delhi": 1.1,
                    "goa": 1.4,
                    "kerala": 1.0,
                    "rajasthan": 1.1,
                    "himachal": 0.9,
                    "default": 1.0,
                }

                # Get destination multiplier
                dest_key = destination.lower()
                multiplier = destination_multipliers.get(
                    dest_key, destination_multipliers["default"]
                )

                # Get base costs
                costs = base_costs.get(accommodation_level, base_costs["mid-range"])

                # Calculate total budget
                daily_cost_per_person = sum(costs.values()) * multiplier
                total_daily_cost = daily_cost_per_person * traveler_count
                total_trip_cost = total_daily_cost * duration_days

                # Add contingency
                contingency = total_trip_cost * 0.15  # 15% buffer
                total_with_contingency = total_trip_cost + contingency

                budget_breakdown = {
                    "destination": destination,
                    "duration_days": duration_days,
                    "traveler_count": traveler_count,
                    "accommodation_level": accommodation_level,
                    "daily_cost_breakdown": {
                        category: cost * multiplier * traveler_count
                        for category, cost in costs.items()
                    },
                    "daily_total": total_daily_cost,
                    "trip_total": total_trip_cost,
                    "contingency": contingency,
                    "recommended_budget": total_with_contingency,
                    "budget_ranges": {
                        "minimum": total_trip_cost * 0.8,
                        "comfortable": total_with_contingency,
                        "luxury": total_with_contingency * 1.5,
                    },
                    "cost_optimization_tips": [
                        "Book accommodations in advance for better rates",
                        "Consider local transportation options",
                        "Mix of paid and free activities",
                        "Try local cuisine for authentic and affordable meals",
                    ],
                }

                return budget_breakdown

        metadata = ToolMetadata(
            name="calculate_trip_budget",
            description="Calculate comprehensive trip budget with breakdown and optimization tips",
            category=ToolCategory.TRIP_PLANNING,
            parameters=[
                ToolParameter(
                    name="destination",
                    type_hint=str,
                    description="Trip destination",
                    required=True,
                    example="Mumbai, India",
                ),
                ToolParameter(
                    name="duration_days",
                    type_hint=int,
                    description="Trip duration in days",
                    required=True,
                    example=5,
                ),
                ToolParameter(
                    name="traveler_count",
                    type_hint=int,
                    description="Number of travelers",
                    required=False,
                    default=1,
                    example=2,
                ),
                ToolParameter(
                    name="trip_type",
                    type_hint=str,
                    description="Type of trip (leisure, business, adventure)",
                    required=False,
                    default="leisure",
                    example="leisure",
                ),
                ToolParameter(
                    name="accommodation_level",
                    type_hint=str,
                    description="Accommodation level (budget, mid-range, luxury)",
                    required=False,
                    default="mid-range",
                    example="mid-range",
                ),
            ],
            chaining_compatible=True,
            cacheable=True,
            examples=[
                {
                    "destination": "Goa",
                    "duration_days": 7,
                    "traveler_count": 2,
                    "accommodation_level": "mid-range",
                },
            ],
            tags=["budget", "planning", "cost", "trip"],
        )

        return BudgetCalculatorTool(metadata)

    @staticmethod
    def create_itinerary_optimizer_tool() -> EnhancedFunctionTool:
        """Create itinerary optimization tool."""

        class ItineraryOptimizerTool(EnhancedFunctionTool):
            async def execute(self, **kwargs) -> Dict[str, Any]:
                validated_params = self.validate_parameters(kwargs)

                validated_params["itinerary_data"]
                optimization_criteria = validated_params.get(
                    "optimization_criteria", ["time", "cost"]
                )
                validated_params.get("preferences", {})

                # Mock optimization logic (in production, use advanced algorithms)
                optimizations = {
                    "time_optimizations": [
                        "Group nearby attractions by location",
                        "Optimize travel routes to minimize commute time",
                        "Schedule indoor activities during peak heat hours",
                        "Plan early morning visits for popular attractions",
                    ],
                    "cost_optimizations": [
                        "Combine attraction tickets for group discounts",
                        "Use public transport instead of private taxis",
                        "Schedule free walking tours and city parks",
                        "Look for lunch specials at local restaurants",
                    ],
                    "experience_optimizations": [
                        "Balance cultural, adventure, and relaxation activities",
                        "Include local experiences and hidden gems",
                        "Schedule buffer time for spontaneous discoveries",
                        "Mix guided tours with independent exploration",
                    ],
                }

                # Generate optimized recommendations
                selected_optimizations = []
                for criterion in optimization_criteria:
                    if criterion in optimizations:
                        selected_optimizations.extend(optimizations[criterion])

                # Mock itinerary scoring
                optimization_score = {
                    "overall_score": 8.5,
                    "time_efficiency": 9.0,
                    "cost_effectiveness": 8.0,
                    "experience_quality": 8.5,
                    "feasibility": 9.0,
                }

                return {
                    "optimized_itinerary": {
                        "optimization_applied": True,
                        "criteria_used": optimization_criteria,
                        "recommendations": selected_optimizations,
                        "estimated_improvements": {
                            "time_saved": "2-3 hours per day",
                            "cost_reduction": "15-25%",
                            "experience_enhancement": "Significant",
                        },
                    },
                    "optimization_score": optimization_score,
                    "implementation_notes": [
                        "Book time-sensitive activities in advance",
                        "Keep contact details for local guides",
                        "Download offline maps for navigation",
                        "Have backup plans for weather-dependent activities",
                    ],
                }

        metadata = ToolMetadata(
            name="optimize_itinerary",
            description="Optimize trip itinerary for time, cost, and experience quality",
            category=ToolCategory.OPTIMIZATION,
            parameters=[
                ToolParameter(
                    name="itinerary_data",
                    type_hint=dict,
                    description="Current itinerary data to optimize",
                    required=True,
                ),
                ToolParameter(
                    name="optimization_criteria",
                    type_hint=list,
                    description="Optimization criteria (time, cost, experience)",
                    required=False,
                    default=["time", "cost"],
                    example=["time", "cost", "experience"],
                ),
                ToolParameter(
                    name="preferences",
                    type_hint=dict,
                    description="User preferences and constraints",
                    required=False,
                    default={},
                ),
            ],
            chaining_compatible=True,
            examples=[
                {"itinerary_data": {}, "optimization_criteria": ["time", "cost"]}
            ],
            tags=["optimization", "itinerary", "planning", "efficiency"],
        )

        return ItineraryOptimizerTool(metadata)


class WeatherEventFramework:
    """Framework for weather and events integration (prepared for external APIs)."""

    @staticmethod
    def create_weather_info_tool() -> EnhancedFunctionTool:
        """Create weather information tool (framework for external API integration)."""

        class WeatherInfoTool(EnhancedFunctionTool):
            async def execute(self, **kwargs) -> Dict[str, Any]:
                validated_params = self.validate_parameters(kwargs)

                location = validated_params["location"]
                date_range = validated_params.get("date_range")

                # Mock weather data (prepare for real API integration)
                mock_weather = {
                    "location": location,
                    "date_range": date_range,
                    "current_weather": {
                        "temperature": "28°C",
                        "condition": "Partly Cloudy",
                        "humidity": "65%",
                        "wind_speed": "12 km/h",
                    },
                    "forecast": [
                        {
                            "date": "2024-01-15",
                            "high": "30°C",
                            "low": "22°C",
                            "condition": "Sunny",
                        },
                        {
                            "date": "2024-01-16",
                            "high": "31°C",
                            "low": "23°C",
                            "condition": "Partly Cloudy",
                        },
                        {
                            "date": "2024-01-17",
                            "high": "29°C",
                            "low": "21°C",
                            "condition": "Light Rain",
                        },
                    ],
                    "travel_recommendations": [
                        "Pack light rain jacket for potential showers",
                        "Comfortable walking shoes recommended",
                        "Sun protection advised for outdoor activities",
                    ],
                    "api_integration_ready": True,
                    "data_source": "Mock Data - Ready for Weather API",
                }

                return mock_weather

        metadata = ToolMetadata(
            name="get_weather_info",
            description="Get weather information for travel planning (framework ready)",
            category=ToolCategory.WEATHER,
            parameters=[
                ToolParameter(
                    name="location",
                    type_hint=str,
                    description="Location for weather information",
                    required=True,
                    example="Mumbai, India",
                ),
                ToolParameter(
                    name="date_range",
                    type_hint=str,
                    description="Date range for forecast (YYYY-MM-DD to YYYY-MM-DD)",
                    required=False,
                    example="2024-01-15 to 2024-01-20",
                ),
            ],
            chaining_compatible=True,
            cacheable=True,
            tags=["weather", "forecast", "travel", "planning"],
        )

        return WeatherInfoTool(metadata)

    @staticmethod
    def create_events_info_tool() -> EnhancedFunctionTool:
        """Create events information tool (framework for external API integration)."""

        class EventsInfoTool(EnhancedFunctionTool):
            async def execute(self, **kwargs) -> Dict[str, Any]:
                validated_params = self.validate_parameters(kwargs)

                location = validated_params["location"]
                date_range = validated_params.get("date_range")
                validated_params.get("event_types", ["cultural", "entertainment"])

                # Mock events data (prepare for real API integration)
                mock_events = {
                    "location": location,
                    "date_range": date_range,
                    "events": [
                        {
                            "name": "Mumbai Cultural Festival",
                            "date": "2024-01-16",
                            "type": "cultural",
                            "location": "NCPA, Nariman Point",
                            "description": "Traditional music and dance performances",
                            "ticket_required": True,
                        },
                        {
                            "name": "Street Food Festival",
                            "date": "2024-01-17",
                            "type": "culinary",
                            "location": "Bandra West",
                            "description": "Local street food vendors and cooking demos",
                            "ticket_required": False,
                        },
                    ],
                    "recommendations": [
                        "Book cultural event tickets in advance",
                        "Check local event calendars for updates",
                        "Consider seasonal festivals and celebrations",
                    ],
                    "api_integration_ready": True,
                    "data_source": "Mock Data - Ready for Events API",
                }

                return mock_events

        metadata = ToolMetadata(
            name="get_local_events",
            description="Get local events information for travel planning (framework ready)",
            category=ToolCategory.EVENTS,
            parameters=[
                ToolParameter(
                    name="location",
                    type_hint=str,
                    description="Location for events search",
                    required=True,
                    example="Mumbai, India",
                ),
                ToolParameter(
                    name="date_range",
                    type_hint=str,
                    description="Date range for events (YYYY-MM-DD to YYYY-MM-DD)",
                    required=False,
                    example="2024-01-15 to 2024-01-20",
                ),
                ToolParameter(
                    name="event_types",
                    type_hint=list,
                    description="Types of events to search for",
                    required=False,
                    default=["cultural", "entertainment"],
                    example=["cultural", "sports", "entertainment"],
                ),
            ],
            chaining_compatible=True,
            cacheable=True,
            tags=["events", "local", "activities", "entertainment"],
        )

        return EventsInfoTool(metadata)


class ToolChainOrchestrator:
    """Orchestrator for chaining and coordinating multiple tools."""

    def __init__(self, tool_registry) -> None:
        self.tool_registry = tool_registry
        self._chain_execution_history: List[Dict[str, Any]] = []

    async def execute_tool_chain(
        self,
        chain_definition: List[Dict[str, Any]],
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a chain of tools with context passing."""

        chain_id = str(uuid4())[:8]
        start_time = datetime.now(timezone.utc)
        context = initial_context or {}

        chain_results = {
            "chain_id": chain_id,
            "start_time": start_time.isoformat(),
            "steps": [],
            "final_context": context.copy(),
            "success": True,
            "total_execution_time": 0.0,
        }

        try:
            for i, step in enumerate(chain_definition):
                step_start = datetime.now(timezone.utc)

                tool_name = step["tool"]
                parameters = step.get("parameters", {})
                context_mapping = step.get("context_mapping", {})

                # Apply context mapping
                for param_key, context_key in context_mapping.items():
                    if context_key in context:
                        parameters[param_key] = context[context_key]

                # Execute tool
                try:
                    result = await self.tool_registry.execute_tool(
                        tool_name, parameters
                    )

                    step_duration = (
                        datetime.now(timezone.utc) - step_start
                    ).total_seconds()

                    step_result = {
                        "step_index": i,
                        "tool_name": tool_name,
                        "parameters": parameters,
                        "result": result,
                        "execution_time": step_duration,
                        "success": True,
                    }

                    # Update context with results
                    if isinstance(result.get("result"), dict):
                        context.update(result["result"])

                    chain_results["steps"].append(step_result)

                except Exception as e:
                    step_duration = (
                        datetime.now(timezone.utc) - step_start
                    ).total_seconds()

                    step_result = {
                        "step_index": i,
                        "tool_name": tool_name,
                        "parameters": parameters,
                        "error": str(e),
                        "execution_time": step_duration,
                        "success": False,
                    }

                    chain_results["steps"].append(step_result)
                    chain_results["success"] = False

                    # Check if step is critical
                    if step.get("critical", True):
                        break

            end_time = datetime.now(timezone.utc)
            chain_results["end_time"] = end_time.isoformat()
            chain_results["total_execution_time"] = (
                end_time - start_time
            ).total_seconds()
            chain_results["final_context"] = context

            # Store execution history
            self._chain_execution_history.append(
                {
                    "chain_id": chain_id,
                    "timestamp": start_time.isoformat(),
                    "success": chain_results["success"],
                    "duration": chain_results["total_execution_time"],
                    "steps_count": len(chain_definition),
                    "successful_steps": sum(
                        1 for step in chain_results["steps"] if step["success"]
                    ),
                }
            )

            return chain_results

        except Exception as e:
            chain_results["error"] = str(e)
            chain_results["success"] = False
            chain_results["end_time"] = datetime.now(timezone.utc).isoformat()
            return chain_results

    def get_chain_analytics(self) -> Dict[str, Any]:
        """Get analytics for tool chain executions."""
        if not self._chain_execution_history:
            return {"message": "No chain execution history available"}

        successful_chains = [c for c in self._chain_execution_history if c["success"]]

        return {
            "total_chains": len(self._chain_execution_history),
            "successful_chains": len(successful_chains),
            "success_rate": (
                len(successful_chains) / len(self._chain_execution_history)
            )
            * 100,
            "average_duration": statistics.mean(
                [c["duration"] for c in self._chain_execution_history]
            ),
            "average_steps": statistics.mean(
                [c["steps_count"] for c in self._chain_execution_history]
            ),
            "recent_executions": self._chain_execution_history[-10:],
        }


class ToolRegistry:
    """Tool registry for backward compatibility."""

    def __init__(self) -> None:
        self._tools = {}

    def register_tool(self, tool):
        """Register a tool."""
        metadata = getattr(tool, "metadata", None)
        if metadata is None or not getattr(metadata, "name", None):
            msg = "Tool must expose metadata with a name field"
            raise ValueError(msg)

        self._tools[tool.metadata.name] = tool

    def get_tool(self, name: str):
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tool_stats(self) -> Dict[str, Any]:
        """Get basic statistics about registered tools."""
        return {
            "total_tools": len(self._tools),
            "tool_categories": {},
            "tool_names": list(self._tools.keys()),
        }

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise FunctionCallError(f"Tool '{tool_name}' not found")

        if hasattr(tool, "execute"):
            if asyncio.iscoroutinefunction(tool.execute):
                return await tool.execute(**kwargs)
            else:
                return tool.execute(**kwargs)
        else:
            raise FunctionCallError(f"Tool '{tool_name}' does not have execute method")


# Enhanced ToolRegistry class:
class EnhancedToolRegistry(ToolRegistry):
    """Enhanced tool registry with analytics and orchestration capabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.chain_orchestrator = ToolChainOrchestrator(self)
        self._register_enhanced_tools()

    def _register_enhanced_tools(self) -> None:
        """Register enhanced and trip planning tools."""
        try:
            # Register trip planning tools
            trip_tools = TripPlanningTools()
            self.register_tool(trip_tools.create_budget_calculator_tool())
            self.register_tool(trip_tools.create_itinerary_optimizer_tool())

            # Register weather and events framework tools
            weather_framework = WeatherEventFramework()
            self.register_tool(weather_framework.create_weather_info_tool())
            self.register_tool(weather_framework.create_events_info_tool())

            logger.info("Enhanced tools registered successfully")

        except Exception:
            logger.exception("Failed to register enhanced tools")

    async def execute_tool_chain(
        self,
        chain_definition: List[Dict[str, Any]],
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a chain of tools."""
        return await self.chain_orchestrator.execute_tool_chain(
            chain_definition, initial_context
        )

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_enhanced_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for tools and chains."""
        base_stats = self.get_tool_stats()

        # Add enhanced analytics
        enhanced_tools = [
            tool
            for tool in self._tools.values()
            if isinstance(tool, EnhancedFunctionTool)
        ]

        tool_analytics = {}
        for tool in enhanced_tools:
            tool_analytics[tool.metadata.name] = tool.get_analytics()

        chain_analytics = self.chain_orchestrator.get_chain_analytics()

        return {
            **base_stats,
            "enhanced_tools_count": len(enhanced_tools),
            "tool_analytics": tool_analytics,
            "chain_analytics": chain_analytics,
            "performance_summary": {
                "avg_tool_performance": (
                    statistics.mean(
                        [
                            analytics.get("avg_execution_time", 0)
                            for analytics in tool_analytics.values()
                        ]
                    )
                    if tool_analytics
                    else 0
                ),
                "overall_success_rate": (
                    statistics.mean(
                        [
                            analytics.get("success_rate", 100)
                            for analytics in tool_analytics.values()
                        ]
                    )
                    if tool_analytics
                    else 100
                ),
            },
        }


# Update the global registry to support both original and enhanced:
_enhanced_tool_registry: Optional[EnhancedToolRegistry] = None


def get_enhanced_tool_registry() -> EnhancedToolRegistry:
    """Get enhanced global tool registry instance."""
    global _enhanced_tool_registry

    if _enhanced_tool_registry is None:
        _enhanced_tool_registry = EnhancedToolRegistry()

    return _enhanced_tool_registry


# Add convenience function for creating trip planning tool chains:
def create_comprehensive_trip_planning_chain(
    destination: str, duration_days: int, _budget: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Create a comprehensive trip planning tool chain."""

    chain = [
        {
            "tool": "calculate_trip_budget",
            "parameters": {
                "destination": destination,
                "duration_days": duration_days,
                "accommodation_level": "mid-range",
            },
            "context_mapping": {},
            "critical": True,
        },
        {
            "tool": "get_weather_info",
            "parameters": {"location": destination},
            "context_mapping": {},
            "critical": False,
        },
        {
            "tool": "get_local_events",
            "parameters": {"location": destination},
            "context_mapping": {},
            "critical": False,
        },
        {
            "tool": "find_places",
            "parameters": {"query": f"tourist attractions in {destination}"},
            "context_mapping": {},
            "critical": True,
        },
        {
            "tool": "optimize_itinerary",
            "parameters": {"optimization_criteria": ["time", "cost", "experience"]},
            "context_mapping": {"itinerary_data": "places"},
            "critical": False,
        },
    ]

    return chain


# Global tool registry instance - keep original intact
_tool_registry: Optional["ToolRegistry"] = None


def get_tool_registry() -> "ToolRegistry":
    """Get global tool registry instance.

    Returns:
        ToolRegistry: Tool registry instance
    """
    global _tool_registry

    if _tool_registry is None:
        from src.ai_services.function_tools import ToolRegistry

        _tool_registry = ToolRegistry()

    return _tool_registry


def tool_function(
    name: str,
    description: str,
    category: ToolCategory,
    parameters: Optional[List[ToolParameter]] = None,
    **metadata_kwargs,
) -> Callable:
    """Decorator to register a function as a tool.

    Args:
        name: Tool name
        description: Tool description
        category: Tool category
        parameters: Tool parameters
        **metadata_kwargs: Additional metadata

    Returns:
        Callable: Decorator function
    """

    def decorator(func: Callable) -> Callable:
        # Auto-detect parameters if not provided
        if parameters is None:
            sig = inspect.signature(func)
            auto_parameters = []

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                auto_parameters.append(
                    ToolParameter(
                        name=param_name,
                        type_hint=(
                            param.annotation
                            if param.annotation != inspect.Parameter.empty
                            else str
                        ),
                        description=f"Parameter {param_name}",
                        required=param.default == inspect.Parameter.empty,
                        default=(
                            param.default
                            if param.default != inspect.Parameter.empty
                            else None
                        ),
                    )
                )

            tool_parameters = auto_parameters
        else:
            tool_parameters = parameters

        # Create metadata
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            parameters=tool_parameters,
            **metadata_kwargs,
        )

        # Create tool class
        class DecoratedTool(FunctionTool):
            async def execute(self, **kwargs) -> Any:
                validated_params = self.validate_parameters(kwargs)

                if inspect.iscoroutinefunction(func):
                    return await func(**validated_params)
                else:
                    return await asyncio.to_thread(func, **validated_params)

        # Register tool
        tool_instance = DecoratedTool(metadata)
        get_tool_registry().register_tool(tool_instance)

        @wraps(func)
        async def wrapper(**kwargs):
            return await tool_instance.execute(**kwargs)

        wrapper._tool_instance = tool_instance  # type: ignore
        return wrapper

    return decorator


# Cleanup function
async def cleanup_tools() -> None:
    """Clean up external API connections."""
    registry = get_tool_registry()

    for tool in registry._tools.values():
        if isinstance(tool, ExternalAPITool):
            try:
                await tool.close()
            except Exception as e:
                logger.warning(f"Error closing tool {tool.metadata.name}: {e}")

    logger.info("Tool cleanup completed")

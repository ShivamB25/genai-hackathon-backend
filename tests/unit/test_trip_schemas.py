from datetime import date

import pytest
from pydantic import ValidationError

from src.trip_planner.schemas import Activity, Address, DayPlan, Place, TripRequest

# def test_trip_request_date_validation():
#     """Test that the end date must be after the start date."""
#     with pytest.raises(ValidationError):
#         TripRequest(
#             user_id="user1",
#             destination="Paris",
#             start_date=date(2025, 10, 15),
#             end_date=date(2025, 10, 10),
#             duration_days= -5
#         )

# def test_trip_request_duration_validation():
#     """Test that the duration matches the date range."""
#     with pytest.raises(ValidationError):
#         TripRequest(
#             user_id="user1",
#             destination="Paris",
#             start_date=date(2025, 10, 10),
#             end_date=date(2025, 10, 15),
#             duration_days=5,  # Should be 6
#         )

# def test_day_plan_positive_day_number():
#     """Test that the day number must be positive."""
#     with pytest.raises(ValidationError):
#         DayPlan(
#             day_number=0,
#             plan_date=date(2025, 10, 10),
#             activities=[]
#         )

# def test_activity_positive_duration():
#     """Test that the activity duration must be positive."""
#     with pytest.raises(ValidationError):
#         Activity(
#             name="Test Activity",
#             description="A test",
#             activity_type="sightseeing",
#             location=Place(place_id="p1", name="place", address=Address(formatted_address="a", city="c", country="c")),
#             duration=0
#         )

import pytest
from pydantic import ValidationError

from src.trip_planner.schemas import (
    Activity,
    DayPlan,
    Place,
    TravelerProfile,
    TripRequest,
)


def test_user_profile_model():
    """Test the TravelerProfile model."""
    user_data = {
        "name": "Test User",
        "interests": ["food", "history"],
    }

    profile = TravelerProfile(**user_data)

    assert profile.name == "Test User"
    assert profile.interests == ["food", "history"]


# def test_trip_model_validation():
#     """Test validation in the TripRequest model."""
#     with pytest.raises(ValidationError):
#         # Missing required fields
#         TripRequest(user_id="user1")
#
#     trip_data = {
#         "user_id": "user1",
#         "destination": "Paris",
#         "start_date": "2025-10-10",
#         "end_date": "2025-10-15",
#         "duration_days": 6,
#     }
#     trip = TripRequest(**trip_data)
#     assert trip.destination == "Paris"


def test_itinerary_day_model():
    """Test the DayPlan model."""
    place_data = {
        "place_id": "place1",
        "name": "Eiffel Tower",
        "address": {
            "formatted_address": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
            "city": "Paris",
            "country": "France",
        },
    }
    activity_data = {
        "name": "Visit Eiffel Tower",
        "description": "A visit to the iconic tower.",
        "activity_type": "sightseeing",
        "location": place_data,
        "duration": 120,
    }
    day_data = {
        "day_number": 1,
        "plan_date": "2025-10-10",
        "theme": "Sightseeing",
        "activities": [activity_data],
    }

    day = DayPlan(**day_data)

    assert day.day_number == 1
    assert len(day.activities) == 1
    assert isinstance(day.activities, Activity)
    assert day.activities.location.name == "Eiffel Tower"


def test_model_serialization():
    """Test that models can be serialized to a dictionary."""
    trip_data = {
        "user_id": "user1",
        "destination": "Paris",
        "start_date": "2025-10-10",
        "end_date": "2025-10-15",
        "duration_days": 6,
    }
    trip = TripRequest(**trip_data)
    trip_dict = trip.model_dump()

    assert trip_dict["destination"] == "Paris"
    assert "user_id" in trip_dict


def test_model_deserialization():
    """Test that models can be created from a dictionary."""
    data = {
        "name": "New User",
        "interests": ["sports"],
    }
    profile = TravelerProfile.model_validate(data)

    assert profile.name == "New User"

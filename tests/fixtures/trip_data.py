VALID_TRIP_REQUEST = {
    "destination": "Maui, Hawaii",
    "start_date": "2025-12-20",
    "end_date": "2025-12-27",
    "user_preferences": ["beaches", "hiking", "snorkeling"],
    "travelers": [
        {"name": "Alice", "age": 30, "travel_style": "adventurous"},
        {"name": "Bob", "age": 32, "travel_style": "relaxed"},
    ],
    "budget": {"min": 3000, "max": 5000, "currency": "USD"},
}

INVALID_TRIP_REQUEST = {
    "destination": "Maui, Hawaii",
    "start_date": "2025-12-27",
    "end_date": "2025-12-20",  # Invalid date range
    "user_preferences": [],
}

MOCK_SIMPLE_AI_RESPONSE = "This is a simple response from the AI model."

MOCK_AI_RESPONSE_WITH_FUNCTION_CALL = """
Here is some information about your trip.

call_function("find_places", {"query": "Eiffel Tower"})
"""

MOCK_STRUCTURED_TRIP_PLAN = {
    "trip_name": "Paris Adventure",
    "days": [
        {
            "day": 1,
            "theme": "Arrival and Exploration",
            "activities": [
                {"time": "14:00", "description": "Arrive at CDG airport"},
                {"time": "16:00", "description": "Check into hotel"},
                {"time": "18:00", "description": "Dinner at a local bistro"},
            ],
        }
    ],
}

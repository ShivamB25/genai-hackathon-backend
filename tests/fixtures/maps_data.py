MOCK_PLACES_RESPONSE = {
    "results": [
        {
            "name": "Eiffel Tower",
            "place_id": "ChIJLU7jZ19x5kcR__z_jE4-xw",
            "rating": 4.6,
            "user_ratings_total": 12345,
            "vicinity": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
        }
    ]
}

MOCK_EMPTY_PLACES_RESPONSE = {"results": []}

MOCK_GEOCODING_RESPONSE = {
    "results": [
        {
            "formatted_address": "Paris, France",
            "geometry": {"location": {"lat": 48.8566, "lng": 2.3522}},
        }
    ]
}

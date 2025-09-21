# API Reference

> Complete API documentation for the AI-Powered Trip Planner Backend

## Table of Contents

- [Authentication](#authentication)
- [Trip Planning API](#trip-planning-api)
- [Maps & Places API](#maps--places-api)
- [User Management API](#user-management-api)
- [Health & Monitoring API](#health--monitoring-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [OpenAPI Integration](#openapi-integration)

---

## Authentication

The API uses Firebase Authentication for secure user management. All protected endpoints require a valid Firebase ID token.

### Authentication Header

```http
Authorization: Bearer <firebase_id_token>
```

### Getting Firebase ID Token

```javascript
// Frontend JavaScript example
import { getAuth, getIdToken } from 'firebase/auth';

const auth = getAuth();
const user = auth.currentUser;
const idToken = await getIdToken(user);

// Use in API requests
fetch('/api/v1/trips/plan', {
  headers: {
    'Authorization': `Bearer ${idToken}`,
    'Content-Type': 'application/json'
  }
});
```

### Authentication Errors

| Code | Message | Description |
|------|---------|-------------|
| 401 | Invalid or expired token | Token is missing, invalid, or expired |
| 403 | Insufficient permissions | User lacks required permissions |
| 429 | Rate limit exceeded | Too many requests from user |

---

## Trip Planning API

Base path: `/api/v1/trips`

### Create AI-Generated Trip Plan

Generate a comprehensive trip itinerary using the multi-agent AI system.

```http
POST /api/v1/trips/plan
```

**Headers:**
- `Authorization: Bearer <token>` (required)
- `Content-Type: application/json`

**Query Parameters:**
- `workflow_type` (string, optional): Workflow type - `comprehensive` (default) or `quick`
- `background` (boolean, optional): Execute as background task - default `false`

**Request Body:**

```json
{
  "destination": "Kerala, India",
  "additional_destinations": ["Munnar", "Alleppey"],
  "start_date": "2024-03-15",
  "end_date": "2024-03-22",
  "traveler_count": 2,
  "trip_type": "leisure",
  "budget_amount": 50000.00,
  "budget_currency": "INR",
  "preferred_activities": ["cultural", "nature", "relaxation"],
  "accommodation_preferences": {
    "type": "hotel",
    "star_rating": 4,
    "amenities": ["wifi", "pool", "spa"]
  },
  "transportation_preferences": ["flight", "car", "boat"],
  "special_requirements": ["vegetarian_food", "air_conditioning"],
  "accessibility_needs": ["wheelchair_accessible"],
  "dietary_restrictions": ["vegetarian"],
  "must_include": ["backwaters_cruise", "spice_plantation"],
  "avoid": ["crowded_places", "expensive_restaurants"]
}
```

**Response (201 Created):**

```json
{
  "workflow_id": "wf_123456789",
  "execution_id": "exec_987654321",
  "request_id": "req_456789123",
  "success": true,
  "itinerary": {
    "itinerary_id": "itin_789123456",
    "title": "7-Day Kerala Cultural & Nature Experience",
    "description": "A perfect blend of cultural immersion and natural beauty",
    "destination": "Kerala, India",
    "start_date": "2024-03-15",
    "end_date": "2024-03-22",
    "duration_days": 7,
    "traveler_count": 2,
    "daily_plans": [
      {
        "day_number": 1,
        "plan_date": "2024-03-15",
        "theme": "Arrival & Kochi Exploration",
        "activities": [
          {
            "activity_id": "act_001",
            "name": "Fort Kochi Walking Tour",
            "description": "Explore colonial architecture and Chinese fishing nets",
            "activity_type": "cultural",
            "location": {
              "place_id": "ChIJbU60yXjnAzsR4jF4tHe_VpU",
              "name": "Fort Kochi",
              "address": {
                "formatted_address": "Fort Kochi, Kochi, Kerala, India",
                "city": "Kochi",
                "country": "India",
                "location": {
                  "latitude": 9.9647,
                  "longitude": 76.2397
                }
              },
              "rating": 4.3,
              "price_level": "budget"
            },
            "start_time": "2024-03-15T09:00:00Z",
            "end_time": "2024-03-15T12:00:00Z",
            "duration": 180,
            "cost": 1500.00,
            "currency": "INR"
          }
        ],
        "transportation": [
          {
            "transport_id": "trans_001",
            "mode": "flight",
            "origin": "Mumbai, India",
            "destination": "Kochi, Kerala",
            "departure_time": "2024-03-15T06:00:00Z",
            "arrival_time": "2024-03-15T08:30:00Z",
            "duration": 150,
            "cost": 8500.00,
            "currency": "INR"
          }
        ],
        "accommodation": {
          "place_id": "hotel_001",
          "name": "Spice Heritage Hotel",
          "address": {
            "formatted_address": "Princess Street, Fort Kochi, Kerala",
            "city": "Kochi",
            "country": "India"
          },
          "rating": 4.2,
          "price_level": "moderate"
        },
        "total_cost": 12000.00,
        "estimated_walking": 3.5,
        "notes": ["Carry sunscreen", "Comfortable walking shoes recommended"]
      }
    ],
    "overall_budget": {
      "total_budget": 50000.00,
      "currency": "INR",
      "breakdown": {
        "accommodation": 21000.00,
        "activities": 8500.00,
        "transportation": 15000.00,
        "meals": 5500.00
      },
      "remaining_amount": 0.00,
      "daily_budget": 7142.86
    },
    "packing_suggestions": [
      "Lightweight cotton clothing",
      "Comfortable walking shoes",
      "Sunscreen and hat",
      "Insect repellent"
    ],
    "local_customs": [
      "Remove shoes before entering temples",
      "Dress modestly when visiting religious sites",
      "Bargaining is common in local markets"
    ],
    "safety_tips": [
      "Stay hydrated",
      "Use mosquito protection",
      "Keep copies of important documents"
    ]
  },
  "agent_responses": [
    {
      "response_id": "resp_001",
      "agent_id": "dest_expert_kerala",
      "agent_role": "destination_expert",
      "confidence_score": 0.92,
      "execution_time": 45.2,
      "tokens_used": 1250
    }
  ],
  "execution_time": 125.7,
  "agents_used": ["dest_expert_kerala", "budget_advisor_001", "trip_planner_001"],
  "tokens_consumed": 3500,
  "quality_scores": {
    "completeness": 0.95,
    "relevance": 0.88,
    "feasibility": 0.91
  }
}
```

### Get Trip Details

Retrieve detailed information about a specific trip.

```http
GET /api/v1/trips/{trip_id}
```

**Path Parameters:**
- `trip_id` (string, required): Trip identifier

**Response (200 OK):**

```json
{
  "success": true,
  "trip": {
    "itinerary_id": "itin_789123456",
    "title": "7-Day Kerala Cultural & Nature Experience",
    "status": "completed",
    "created_at": "2024-03-10T10:30:00Z",
    "updated_at": "2024-03-12T14:20:00Z"
  },
  "message": "Trip retrieved successfully"
}
```

### Update Trip Itinerary

Update an existing trip with AI optimization.

```http
PUT /api/v1/trips/{trip_id}
```

**Request Body:**

```json
{
  "title": "Updated Kerala Adventure",
  "description": "Modified itinerary with more adventure activities",
  "daily_plans": [
    {
      "day_number": 1,
      "theme": "Adventure Start",
      "activities": []
    }
  ],
  "packing_suggestions": ["Adventure gear", "Waterproof clothing"]
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "trip": {
    "itinerary_id": "itin_789123456",
    "title": "Updated Kerala Adventure",
    "updated_at": "2024-03-12T16:45:00Z"
  },
  "message": "Trip updated successfully"
}
```

### Delete Trip

Remove a trip from the user's collection.

```http
DELETE /api/v1/trips/{trip_id}
```

**Response (200 OK):**

```json
{
  "success": true,
  "message": "Trip deleted successfully",
  "trip_id": "itin_789123456"
}
```

### List User Trips

Get user's trip history with pagination and filtering.

```http
GET /api/v1/trips
```

**Query Parameters:**
- `page` (integer, optional): Page number (default: 1)
- `limit` (integer, optional): Items per page (default: 10, max: 100)
- `destination` (string, optional): Filter by destination
- `status` (string, optional): Filter by status (`draft`, `completed`, `cancelled`)
- `start_date_after` (date, optional): Filter trips starting after date
- `start_date_before` (date, optional): Filter trips starting before date

**Response (200 OK):**

```json
{
  "success": true,
  "trips": [
    {
      "itinerary_id": "itin_789123456",
      "title": "Kerala Cultural Experience",
      "destination": "Kerala, India",
      "start_date": "2024-03-15",
      "duration_days": 7,
      "status": "completed",
      "created_at": "2024-03-10T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 5,
    "has_next_page": false
  },
  "total_count": 5
}
```

### Share Trip

Share a trip with other users.

```http
POST /api/v1/trips/{trip_id}/share
```

**Request Body:**

```json
{
  "user_emails": ["friend@example.com", "family@example.com"]
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "message": "Trip shared successfully",
  "trip_id": "itin_789123456",
  "shared_with": ["friend@example.com", "family@example.com"],
  "updated_at": "2024-03-12T18:30:00Z"
}
```

### Get Trip Metrics

Get comprehensive analytics and metrics for a trip.

```http
GET /api/v1/trips/{trip_id}/metrics
```

**Response (200 OK):**

```json
{
  "success": true,
  "metrics": {
    "total_cost": 48500.00,
    "currency": "INR",
    "activity_distribution": {
      "cultural": 8,
      "nature": 5,
      "adventure": 3,
      "relaxation": 4
    },
    "daily_averages": {
      "activities_per_day": 2.8,
      "cost_per_day": 6928.57,
      "walking_distance": 4.2
    },
    "optimization_score": {
      "budget_efficiency": 0.92,
      "time_optimization": 0.88,
      "route_efficiency": 0.85
    }
  },
  "message": "Trip metrics retrieved successfully"
}
```

### Get Trip Status

Check the status of trip generation (useful for background tasks).

```http
GET /api/v1/trips/{trip_id}/status
```

**Response (200 OK):**

```json
{
  "success": true,
  "status": {
    "trip_id": "itin_789123456",
    "status": "completed",
    "itinerary_available": true,
    "created_at": "2024-03-10T10:30:00Z",
    "updated_at": "2024-03-12T14:20:00Z"
  }
}
```

---

## Maps & Places API

Base path: `/api/v1/places`

### Search Places

Search for places using text queries with location context.

```http
GET /api/v1/places/search
```

**Query Parameters:**
- `query` (string, required): Search query
- `location` (string, optional): Lat,lng for location bias
- `radius` (integer, optional): Search radius in meters (default: 5000, max: 50000)
- `type` (string, optional): Place type filter
- `min_rating` (float, optional): Minimum rating filter (0.0-5.0)
- `price_level` (string, optional): Price level filter (`free`, `budget`, `moderate`, `expensive`, `luxury`)
- `open_now` (boolean, optional): Filter for currently open places
- `language` (string, optional): Response language (default: `en`)

**Response (200 OK):**

```json
{
  "success": true,
  "places": [
    {
      "place_id": "ChIJbU60yXjnAzsR4jF4tHe_VpU",
      "name": "Mattancherry Palace",
      "description": "Historic palace showcasing Kerala's royal heritage",
      "address": {
        "formatted_address": "Mattancherry, Kochi, Kerala 682002, India",
        "street_number": null,
        "street_name": "Palace Road",
        "city": "Kochi",
        "state": "Kerala",
        "country": "India",
        "postal_code": "682002",
        "location": {
          "latitude": 9.9584,
          "longitude": 76.2607
        }
      },
      "contact": {
        "phone_number": "+91-484-2226085",
        "website": "https://www.keralatourism.org/destination/mattancherry-palace-kochi/164"
      },
      "opening_hours": {
        "is_open_now": true,
        "weekday_text": [
          "Monday: Closed",
          "Tuesday: 10:00 AM – 5:00 PM",
          "Wednesday: 10:00 AM – 5:00 PM",
          "Thursday: 10:00 AM – 5:00 PM",
          "Friday: 10:00 AM – 5:00 PM",
          "Saturday: 10:00 AM – 5:00 PM",
          "Sunday: 10:00 AM – 5:00 PM"
        ]
      },
      "rating": 4.1,
      "review_count": 2840,
      "price_level": "budget",
      "place_types": ["tourist_attraction", "museum", "point_of_interest"],
      "photos": [
        "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference=abc123"
      ]
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 45,
    "has_next_page": true
  },
  "search_metadata": {
    "query": "historic palaces kochi",
    "location_bias": "9.9312,76.2673",
    "radius": 10000,
    "execution_time": 0.85
  }
}
```

### Find Nearby Places

Find places near a specific location.

```http
GET /api/v1/places/nearby
```

**Query Parameters:**
- `location` (string, required): Center point as lat,lng
- `radius` (integer, optional): Search radius in meters
- `type` (string, optional): Place type filter
- `keyword` (string, optional): Keyword filter
- `min_rating` (float, optional): Minimum rating filter
- `max_results` (integer, optional): Maximum results (default: 20, max: 60)

**Response (200 OK):**

```json
{
  "success": true,
  "places": [
    {
      "place_id": "ChIJ123abc456def",
      "name": "Spice Market",
      "distance_meters": 450,
      "rating": 4.3,
      "price_level": "budget",
      "place_types": ["store", "food", "point_of_interest"],
      "location": {
        "latitude": 9.9595,
        "longitude": 76.2611
      }
    }
  ],
  "location_context": {
    "center": "9.9584,76.2607",
    "radius": 1000,
    "places_found": 15
  }
}
```

### Get Place Details

Get comprehensive details about a specific place.

```http
GET /api/v1/places/{place_id}
```

**Path Parameters:**
- `place_id` (string, required): Google Place ID

**Query Parameters:**
- `fields` (string, optional): Comma-separated fields to include
- `language` (string, optional): Response language

**Response (200 OK):**

```json
{
  "success": true,
  "place": {
    "place_id": "ChIJbU60yXjnAzsR4jF4tHe_VpU",
    "name": "Mattancherry Palace",
    "description": "Historic palace museum featuring murals and artifacts",
    "address": {
      "formatted_address": "Mattancherry, Kochi, Kerala 682002, India",
      "location": {
        "latitude": 9.9584,
        "longitude": 76.2607
      }
    },
    "contact": {
      "phone_number": "+91-484-2226085",
      "website": "https://www.keralatourism.org/destination/mattancherry-palace-kochi/164",
      "social_media": {
        "facebook": "https://facebook.com/mattancherrypalace"
      }
    },
    "opening_hours": {
      "is_open_now": true,
      "weekday_text": ["Tuesday: 10:00 AM – 5:00 PM"],
      "special_hours": {
        "2024-03-15": "Closed for maintenance"
      }
    },
    "rating": 4.1,
    "review_count": 2840,
    "price_level": "budget",
    "amenities": ["parking", "wheelchair_accessible", "guided_tours"],
    "accessibility": {
      "wheelchair_accessible_entrance": true,
      "wheelchair_accessible_parking": true
    },
    "photos": [
      "https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference=abc123"
    ],
    "reviews": [
      {
        "author_name": "John Doe",
        "rating": 5,
        "text": "Beautiful palace with amazing murals!",
        "time": "2024-02-15T10:30:00Z"
      }
    ],
    "metadata": {
      "last_updated": "2024-03-01T12:00:00Z",
      "data_source": "google_places"
    }
  }
}
```

### Get Directions

Get directions between two or more locations.

```http
POST /api/v1/places/directions
```

**Request Body:**

```json
{
  "origin": "Fort Kochi, Kerala, India",
  "destination": "Munnar, Kerala, India",
  "waypoints": ["Mattancherry Palace", "Marine Drive"],
  "travel_mode": "driving",
  "departure_time": "2024-03-15T09:00:00Z",
  "traffic_model": "best_guess",
  "avoid": ["tolls", "highways"],
  "optimize_waypoints": true,
  "language": "en",
  "units": "metric"
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "routes": [
    {
      "route_id": "route_001",
      "summary": "Optimal route via NH85",
      "distance": {
        "text": "130 km",
        "value": 130000
      },
      "duration": {
        "text": "4 hours 15 mins",
        "value": 15300
      },
      "duration_in_traffic": {
        "text": "4 hours 45 mins",
        "value": 17100
      },
      "start_location": {
        "latitude": 9.9647,
        "longitude": 76.2397
      },
      "end_location": {
        "latitude": 10.0889,
        "longitude": 77.0595
      },
      "legs": [
        {
          "distance": {
            "text": "45 km",
            "value": 45000
          },
          "duration": {
            "text": "1 hour 30 mins",
            "value": 5400
          },
          "start_location": {
            "latitude": 9.9647,
            "longitude": 76.2397
          },
          "end_location": {
            "latitude": 9.9584,
            "longitude": 76.2607
          },
          "steps": [
            {
              "distance": {
                "text": "0.5 km",
                "value": 500
              },
              "duration": {
                "text": "2 mins",
                "value": 120
              },
              "html_instructions": "Head <b>southeast</b> on <b>Princess St</b> toward <b>Bazaar Rd</b>",
              "travel_mode": "DRIVING"
            }
          ]
        }
      ],
      "warnings": ["This route includes tolls"],
      "waypoint_order": [0, 1],
      "fare": {
        "currency": "INR",
        "value": 2500,
        "text": "₹2,500"
      }
    }
  ],
  "alternatives": [
    {
      "route_id": "route_002",
      "summary": "Scenic route via backroads",
      "distance": {
        "text": "145 km",
        "value": 145000
      },
      "duration": {
        "text": "5 hours 30 mins",
        "value": 19800
      }
    }
  ],
  "request_metadata": {
    "origin": "Fort Kochi, Kerala, India",
    "destination": "Munnar, Kerala, India",
    "travel_mode": "driving",
    "optimization_applied": true
  }
}
```

### Geocode Location

Convert addresses to coordinates or vice versa.

```http
POST /api/v1/places/geocode
```

**Request Body:**

```json
{
  "address": "Mattancherry Palace, Kochi, Kerala, India",
  "bounds": {
    "northeast": {"lat": 10.0, "lng": 77.0},
    "southwest": {"lat": 9.5, "lng": 76.0}
  },
  "region": "IN",
  "language": "en"
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "results": [
    {
      "formatted_address": "Mattancherry Palace, Mattancherry, Kochi, Kerala 682002, India",
      "geometry": {
        "location": {
          "latitude": 9.9584,
          "longitude": 76.2607
        },
        "location_type": "ROOFTOP",
        "viewport": {
          "northeast": {
            "latitude": 9.9598,
            "longitude": 76.2621
          },
          "southwest": {
            "latitude": 9.9570,
            "longitude": 76.2593
          }
        }
      },
      "address_components": [
        {
          "long_name": "Mattancherry Palace",
          "short_name": "Mattancherry Palace",
          "types": ["establishment", "point_of_interest", "tourist_attraction"]
        },
        {
          "long_name": "Mattancherry",
          "short_name": "Mattancherry",
          "types": ["political", "sublocality", "sublocality_level_1"]
        }
      ],
      "place_id": "ChIJbU60yXjnAzsR4jF4tHe_VpU",
      "types": ["establishment", "point_of_interest", "tourist_attraction"]
    }
  ],
  "geocoding_metadata": {
    "query_type": "address_to_coordinates",
    "execution_time": 0.45,
    "results_count": 1
  }
}
```

---

## User Management API

Base path: `/api/v1/users`

### Get User Profile

Retrieve user profile information from Firestore.

```http
GET /api/v1/users/profile
```

**Response (200 OK):**

```json
{
  "success": true,
  "profile": {
    "uid": "user123",
    "email": "user@example.com",
    "email_verified": true,
    "display_name": "John Doe",
    "first_name": "John",
    "last_name": "Doe",
    "photo_url": "https://example.com/photo.jpg",
    "phone_number": "+919876543210",
    "date_of_birth": "1990-05-15",
    "gender": "male",
    "location": "Mumbai, India",
    "bio": "Travel enthusiast and culture explorer",
    "profile_complete": true,
    "terms_accepted": true,
    "privacy_policy_accepted": true,
    "marketing_emails_enabled": false,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-03-10T14:20:00Z"
  },
  "message": "Profile retrieved successfully"
}
```

### Update User Profile

Update user profile information.

```http
PUT /api/v1/users/profile
```

**Request Body:**

```json
{
  "display_name": "John Smith",
  "first_name": "John",
  "last_name": "Smith",
  "phone_number": "+919876543210",
  "date_of_birth": "1990-05-15",
  "gender": "male",
  "location": "Mumbai, India",
  "bio": "Adventure seeker and nature lover",
  "marketing_emails_enabled": true
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "profile": {
    "uid": "user123",
    "display_name": "John Smith",
    "updated_at": "2024-03-12T16:45:00Z"
  },
  "message": "Profile updated successfully"
}
```

### Get Travel Preferences

Retrieve user's travel preferences.

```http
GET /api/v1/users/preferences
```

**Response (200 OK):**

```json
{
  "preferred_activities": ["cultural", "nature", "adventure"],
  "accommodation_types": ["hotel", "resort"],
  "transportation_modes": ["flight", "train", "car"],
  "budget_range": "moderate",
  "trip_types": ["leisure", "adventure"],
  "dietary_restrictions": ["vegetarian"],
  "accessibility_needs": ["wheelchair_accessible"],
  "language_preferences": ["en", "hi"],
  "currency_preference": "INR",
  "time_zone": "Asia/Kolkata",
  "notification_settings": {
    "trip_updates": true,
    "promotional_offers": false,
    "booking_reminders": true
  }
}
```

### Update Travel Preferences

Update user's travel preferences.

```http
PUT /api/v1/users/preferences
```

**Request Body:**

```json
{
  "preferences": {
    "preferred_activities": ["cultural", "nature", "wellness"],
    "accommodation_types": ["boutique", "resort"],
    "budget_range": "luxury",
    "dietary_restrictions": ["vegan"],
    "notification_settings": {
      "trip_updates": true,
      "promotional_offers": true,
      "booking_reminders": true
    }
  }
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "message": "Travel preferences updated successfully",
  "data": {
    "preferences": {
      "preferred_activities": ["cultural", "nature", "wellness"],
      "budget_range": "luxury"
    }
  }
}
```

### Get User Trip History

Get user's trip history with pagination and filtering.

```http
GET /api/v1/users/trips
```

**Query Parameters:**
- `page` (integer, optional): Page number
- `limit` (integer, optional): Items per page
- `destination` (string, optional): Filter by destination
- `status` (string, optional): Filter by trip status

**Response (200 OK):**

```json
{
  "success": true,
  "trips": [
    {
      "itinerary_id": "itin_789123456",
      "title": "Kerala Cultural Experience",
      "destination": "Kerala, India",
      "start_date": "2024-03-15",
      "duration_days": 7,
      "status": "completed",
      "created_at": "2024-03-10T10:30:00Z"
    }
  ],
  "stats": {
    "trips_created": 5,
    "trips_completed": 3,
    "favorite_destinations": ["Kerala", "Rajasthan", "Goa"],
    "total_distance_traveled": 2500.5,
    "account_age_days": 65,
    "last_activity": "2024-03-12T14:20:00Z"
  },
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 5,
    "has_next_page": false
  }
}
```

### Get User Statistics

Get comprehensive user statistics and activity metrics.

```http
GET /api/v1/users/stats
```

**Response (200 OK):**

```json
{
  "trips_created": 5,
  "trips_completed": 3,
  "favorite_destinations": ["Kerala", "Rajasthan", "Goa"],
  "total_distance_traveled": 2500.5,
  "account_age_days": 65,
  "last_activity": "2024-03-12T14:20:00Z",
  "activity_preferences": {
    "cultural": 12,
    "nature": 8,
    "adventure": 5
  },
  "budget_efficiency": 0.87,
  "average_trip_rating": 4.3
}
```

### Complete Profile Setup

Mark user profile as complete and accept terms.

```http
POST /api/v1/users/complete-profile
```

**Response (200 OK):**

```json
{
  "success": true,
  "message": "Profile setup completed successfully",
  "data": {
    "profile_complete": true
  }
}
```

### Delete User Account

Delete user account and all associated data.

```http
DELETE /api/v1/users/account
```

**Request Body:**

```json
{
  "confirm_deletion": true,
  "reason": "No longer needed"
}
```

**Response (200 OK):**

```json
{
  "success": true,
  "message": "Account deleted successfully",
  "data": {
    "user_id": "user123",
    "deleted_at": "2024-03-12T18:30:00Z",
    "reason": "No longer needed"
  }
}
```

---

## Health & Monitoring API

### System Health Check

Check overall system health and dependencies.

```http
GET /health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "service": "genai-trip-planner",
  "version": "0.1.0",
  "checks": {
    "database": "healthy",
    "firebase": "healthy",
    "vertex_ai": "configured",
    "maps_api": "configured"
  },
  "timestamp": "2024-03-12T12:00:00Z"
}
```

### Liveness Check

Check if the API process is alive (for Kubernetes).

```http
GET /health/live
```

**Response (200 OK):**

```json
{
  "status": "alive",
  "service": "genai-trip-planner",
  "version": "0.1.0"
}
```

### Trip Planning Service Health

```http
GET /api/v1/trips/health
```

**Response (200 OK):**

```json
{
  "success": true,
  "health": {
    "status": "healthy",
    "service": "trip_planning",
    "dependencies": {
      "ai_orchestrator": "healthy",
      "firestore": "healthy",
      "maps_services": "healthy"
    },
    "timestamp": "2024-03-12T12:00:00Z"
  }
}
```

### User Services Health

```http
GET /api/v1/users/health
```

**Response (200 OK):**

```json
{
  "success": true,
  "health": {
    "status": "healthy",
    "services": {
      "firebase_auth": "healthy",
      "firestore_users": "healthy",
      "profile_management": "healthy"
    },
    "timestamp": "2024-03-12T12:00:00Z"
  }
}
```

### API Root Information

```http
GET /
```

**Response (200 OK):**

```json
{
  "message": "Welcome to AI-Powered Trip Planner Backend",
  "service": "genai-trip-planner",
  "version": "0.1.0",
  "environment": "development",
  "docs_url": "/docs",
  "health_check": "/health",
  "features": {
    "authentication": "Firebase Auth",
    "database": "Firestore",
    "ai_model": "Vertex AI Gemini Multi-Agent System",
    "maps": "Google Maps API",
    "trip_planning": "AI-Powered Itinerary Generation",
    "background_tasks": "Async Trip Generation & Optimization"
  },
  "api_endpoints": {
    "trip_planning": "/api/v1/trips",
    "places_search": "/api/v1/places",
    "user_management": "/api/v1/users",
    "health_monitoring": "/api/v1/health"
  }
}
```

---

## Error Handling

### Standard Error Response Format

All API errors follow a consistent format:

```json
{
  "detail": "Error description",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-03-12T12:00:00Z",
  "request_id": "req_123456789"
}
```

### HTTP Status Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 502 | Bad Gateway | External service error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

| Error Code | Description | Typical Status |
|------------|-------------|----------------|
| `INVALID_TOKEN` | Invalid or expired Firebase token | 401 |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions | 403 |
| `TRIP_NOT_FOUND` | Trip ID not found | 404 |
| `TRIP_ACCESS_DENIED` | User cannot access trip | 403 |
| `TRIP_GENERATION_FAILED` | AI trip generation failed | 500 |
| `PLACES_API_ERROR` | Google Places API error | 502 |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded | 429 |
| `VALIDATION_ERROR` | Request validation failed | 422 |
| `DATABASE_ERROR` | Firestore operation failed | 500 |
| `AI_SERVICE_UNAVAILABLE` | Vertex AI service unavailable | 503 |

### Error Examples

**Validation Error (422):**

```json
{
  "detail": [
    {
      "loc": ["body", "start_date"],
      "msg": "End date must be after start date",
      "type": "value_error"
    }
  ],
  "error_code": "VALIDATION_ERROR"
}
```

**Rate Limit Error (429):**

```json
{
  "detail": "Rate limit exceeded. Maximum 100 requests per hour per user.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 3600,
  "timestamp": "2024-03-12T12:00:00Z"
}
```

**Trip Generation Error (500):**

```json
{
  "detail": "Trip generation failed: AI service timeout",
  "error_code": "TRIP_GENERATION_FAILED",
  "timestamp": "2024-03-12T12:00:00Z",
  "request_id": "req_123456789"
}
```

---

## Rate Limiting

### Default Limits

| Endpoint Category | Limit | Window |
|------------------|-------|---------|
| Trip Planning | 10 requests | per hour |
| Places Search | 100 requests | per hour |
| User Management | 50 requests | per hour |
| Health Checks | 60 requests | per minute |

### Rate Limit Headers

All responses include rate limiting information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1647086400
X-RateLimit-Window: 3600
```

### Rate Limit Response

When rate limit is exceeded:

```json
{
  "detail": "Rate limit exceeded. Maximum 100 requests per hour per user.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 3600,
  "timestamp": "2024-03-12T12:00:00Z"
}
```

### Custom Rate Limits

Premium users may have higher rate limits. Contact support for custom limits.

---

## OpenAPI Integration

### Interactive Documentation

- **Swagger UI**: `/docs` - Interactive API documentation
- **ReDoc**: `/redoc` - Alternative documentation interface
- **OpenAPI Schema**: `/openapi.json` - OpenAPI 3.0 specification

### API Client Generation

Generate API clients using the OpenAPI specification:

```bash
# Python client
openapi-generator-cli generate -i http://localhost:8000/openapi.json -g python -o ./python-client

# JavaScript client
openapi-generator-cli generate -i http://localhost:8000/openapi.json -g javascript -o ./js-client

# TypeScript client
openapi-generator-cli generate -i http://localhost:8000/openapi.json -g typescript-fetch -o ./ts-client
```

### Postman Collection

Import the API into Postman:

1. Open Postman
2. Click "Import"
3. Enter URL: `http://localhost:8000/openapi.json`
4. Select "Generate collection from OpenAPI 3.0 schema"

### API Versioning

The API uses URL path versioning:

- Current version: `/api/v1/`
- Future versions: `/api/v2/`, `/api/v3/`, etc.

### Content Types

**Supported Request Content Types:**
- `application/json` (default)
- `application/x-www-form-urlencoded`
- `multipart/form-data` (for file uploads)

**Response Content Type:**
- `application/json` (all responses)

### Authentication in OpenAPI

The OpenAPI specification includes Firebase authentication configuration:

```yaml
components:
  securitySchemes:
    FirebaseAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: Firebase ID Token
```

---

## SDK Examples

### Python SDK Example

```python
import asyncio
import httpx
from datetime import datetime, date

class TripPlannerClient:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    async def create_trip_plan(self, **trip_data):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/trips/plan",
                headers=self.headers,
                json=trip_data
            )
            return response.json()

    async def search_places(self, query: str, location: str = None):
        async with httpx.AsyncClient() as client:
            params = {"query": query}
            if location:
                params["location"] = location
                
            response = await client.get(
                f"{self.base_url}/api/v1/places/search",
                headers=self.headers,
                params=params
            )
            return response.json()

# Usage
client = TripPlannerClient("http://localhost:8000", "your_firebase_token")

trip = await client.create_trip_plan(
    destination="Kerala, India",
    start_date="2024-03-15",
    end_date="2024-03-22",
    traveler_count=2,
    budget_amount=50000
)
```

### JavaScript SDK Example

```javascript
class TripPlannerAPI {
  constructor(baseUrl, apiToken) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${apiToken}`,
      'Content-Type': 'application/json'
    };
  }

  async createTripPlan(tripData) {
    const response = await fetch(`${this.baseUrl}/api/v1/trips/plan`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(tripData)
    });
    return response.json();
  }

  async searchPlaces(query, location = null) {
    const params = new URLSearchParams({ query });
    if (location) params.append('location', location);
    
    const response = await fetch(
      `${this.baseUrl}/api/v1/places/search?${params}`,
      { headers: this.headers }
    );
    return response.json();
  }
}

// Usage
const client = new TripPlannerAPI('http://localhost:8000', 'your_firebase_token');

const trip = await client.createTripPlan({
  destination: 'Kerala, India',
  start_date: '2024-03-15',
  end_date: '2024-03-22',
  traveler_count: 2,
  budget_amount: 50000
});
```

---

## Webhooks

### Trip Status Webhooks

Configure webhooks to receive notifications when trip status changes:

**Webhook Payload:**

```json
{
  "event_type": "trip.status_changed",
  "timestamp": "2024-03-12T12:00:00Z",
  "data": {
    "trip_id": "itin_789123456",
    "user_id": "user123",
    "old_status": "generating",
    "new_status": "completed",
    "workflow_execution_id": "exec_987654321"
  },
  "webhook_id": "wh_123456789",
  "api_version": "v1"
}
```

### Webhook Security

All webhook payloads include an HMAC signature for verification:

```http
X-Webhook-Signature: sha256=abc123def456...
X-Webhook-Timestamp: 1647086400
```

---

**For more detailed information, visit the [full documentation](../README.md) or explore the interactive API documentation at `/docs`.**
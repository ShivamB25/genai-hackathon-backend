# AI-Powered Trip Planner Backend

> A comprehensive trip planning backend using Google Vertex AI Gemini, Firebase, and Google Maps APIs to create personalized travel itineraries through intelligent multi-agent workflows.

## 🌟 Features

### 🤖 AI-Powered Trip Planning
- **Multi-Agent System**: Specialized AI agents using Google ADK patterns (Sequential, Parallel, Loop workflows)
- **Intelligent Orchestration**: Dynamic workflow coordination for complex trip planning scenarios
- **Personalized Recommendations**: Context-aware suggestions based on user preferences and real-time data
- **Budget Optimization**: Smart cost analysis and optimization across accommodations, activities, and transportation

### 🗺️ Advanced Maps Integration
- **Google Maps API**: Places search, directions, geocoding, and distance matrix
- **Location Intelligence**: Nearby attractions, restaurants, and points of interest
- **Route Optimization**: Multi-stop journey planning with travel time calculations
- **Real-time Data**: Traffic-aware routing and location-based recommendations

### 🔐 Enterprise Authentication & Security
- **Firebase Authentication**: Secure user management with social login support
- **User Profiles**: Comprehensive preference management and trip history
- **Data Privacy**: GDPR-compliant data handling and user consent management
- **Role-based Access**: Fine-grained permissions for trip sharing and collaboration

### 🎯 Smart Itinerary Generation
- **Day-by-Day Planning**: Structured daily plans with activities, meals, and accommodations
- **Activity Recommendations**: AI-curated suggestions based on interests and location
- **Transportation Planning**: Multi-modal transport optimization and booking integration
- **Weather Integration**: Weather-aware activity suggestions and backup plans

### 📊 Analytics & Optimization
- **Performance Monitoring**: Real-time metrics and health monitoring
- **User Analytics**: Trip preferences analysis and recommendation improvement
- **Cost Tracking**: Detailed budget breakdowns and spending optimization
- **Quality Metrics**: AI response quality assessment and continuous improvement

## 🏗️ Architecture

### System Overview
```mermaid
graph TB
    A[Client Applications] --> B[FastAPI Backend]
    B --> C[Firebase Auth]
    B --> D[Multi-Agent Orchestrator]
    B --> E[Google Maps Services]
    B --> F[Firestore Database]
    
    D --> G[Trip Planner Agent]
    D --> H[Destination Expert Agent]
    D --> I[Budget Advisor Agent]
    D --> J[Itinerary Optimizer Agent]
    
    G --> K[Vertex AI Gemini]
    H --> K
    I --> K
    J --> K
    
    E --> L[Places API]
    E --> M[Directions API]
    E --> N[Geocoding API]
```

### Core Components

| Component | Description | Technology |
|-----------|-------------|------------|
| **API Layer** | FastAPI web framework with OpenAPI documentation | FastAPI, Pydantic |
| **AI Orchestrator** | Multi-agent workflow coordination using Google ADK patterns | Google ADK, Vertex AI |
| **Agent System** | Specialized AI agents for different trip planning aspects | Gemini 2.0 Flash |
| **Maps Services** | Location-based services with caching and rate limiting | Google Maps API |
| **Authentication** | User management and security | Firebase Auth |
| **Database** | Document-based storage for trips and user data | Firestore |
| **Configuration** | Environment-based configuration management | Pydantic Settings |

### Agent Roles
- **Trip Planner**: Main coordination and itinerary creation
- **Destination Expert**: Local knowledge and cultural insights
- **Budget Advisor**: Cost optimization and financial planning
- **Itinerary Optimizer**: Route and schedule optimization
- **Activity Recommender**: Personalized activity suggestions
- **Local Guide**: Insider tips and hidden gems
- **Safety Advisor**: Travel safety and health recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Google Cloud Project with enabled APIs
- Firebase Project
- Google Maps API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd genai-hackathon-backend
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials (see Configuration section)
   ```

4. **Initialize Firebase**
   ```bash
   python scripts/setup-firebase.py
   ```

5. **Start the development server**
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - API Root: http://localhost:8000/

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Application
APP_NAME=genai-trip-planner
DEBUG=true
LOG_LEVEL=INFO
ENVIRONMENT=development

# Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./path/to/service-account.json
VERTEX_AI_REGION=asia-south1

# Firebase
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your-service@your-project.iam.gserviceaccount.com

# Google Maps
GOOGLE_MAPS_API_KEY=your-maps-api-key

# Gemini AI
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=8192

# Security
JWT_SECRET_KEY=your-super-secret-key
ALLOWED_ORIGINS=["http://localhost:3000"]
```

### Google Cloud Setup

1. **Enable required APIs**:
   - Vertex AI API
   - Firebase Admin SDK API
   - Google Maps Platform APIs (Places, Directions, Geocoding)

2. **Create service account**:
   ```bash
   gcloud iam service-accounts create trip-planner-service \
     --display-name="Trip Planner Service Account"
   
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:trip-planner-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   
   gcloud iam service-accounts keys create service-account-key.json \
     --iam-account=trip-planner-service@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

## 📋 API Endpoints

### Trip Planning
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/trips/plan` | Generate AI trip plan |
| GET | `/api/v1/trips/{trip_id}` | Get trip details |
| PUT | `/api/v1/trips/{trip_id}` | Update trip itinerary |
| DELETE | `/api/v1/trips/{trip_id}` | Delete trip |
| GET | `/api/v1/trips` | List user trips |
| POST | `/api/v1/trips/{trip_id}/share` | Share trip with others |

### Places & Maps
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/places/search` | Search for places |
| GET | `/api/v1/places/nearby` | Find nearby places |
| GET | `/api/v1/places/{place_id}` | Get place details |
| POST | `/api/v1/places/directions` | Get directions |
| POST | `/api/v1/places/geocode` | Geocode locations |

### User Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/users/profile` | Get user profile |
| PUT | `/api/v1/users/profile` | Update profile |
| GET | `/api/v1/users/preferences` | Get travel preferences |
| PUT | `/api/v1/users/preferences` | Update preferences |
| GET | `/api/v1/users/trips` | Get trip history |
| DELETE | `/api/v1/users/account` | Delete account |

## 💡 Usage Examples

### Creating a Trip Plan

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/trips/plan",
        headers={"Authorization": "Bearer YOUR_FIREBASE_TOKEN"},
        json={
            "destination": "Kerala, India",
            "start_date": "2024-03-15",
            "end_date": "2024-03-22",
            "traveler_count": 2,
            "trip_type": "leisure",
            "budget_amount": 50000,
            "budget_currency": "INR",
            "preferred_activities": ["cultural", "nature", "relaxation"],
            "special_requirements": ["vegetarian_food", "air_conditioning"]
        }
    )
    
    trip_plan = response.json()
    print(f"Trip ID: {trip_plan['execution_id']}")
```

### Searching Places

```python
response = await client.get(
    "http://localhost:8000/api/v1/places/search",
    params={
        "query": "restaurants in Kochi",
        "location": "9.9312,76.2673",
        "radius": 5000,
        "type": "restaurant"
    }
)

places = response.json()["places"]
for place in places:
    print(f"{place['name']} - Rating: {place['rating']}")
```

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
python src/test_integration.py
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Test trip planning (requires authentication)
curl -X POST http://localhost:8000/api/v1/trips/plan \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Goa, India",
    "start_date": "2024-04-01",
    "end_date": "2024-04-05",
    "traveler_count": 2,
    "budget_amount": 25000
  }'
```

## 🐳 Docker Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
# Build image
docker build -t trip-planner-backend .

# Run container
docker run -p 8000:8000 --env-file .env trip-planner-backend
```

## 📊 Monitoring & Health Checks

### Health Endpoints
- **Main Health**: `/health` - Overall system health
- **Live Check**: `/health/live` - Process liveness
- **Trip Service**: `/api/v1/trips/health` - Trip planning service health
- **User Service**: `/api/v1/users/health` - User management health

### Metrics
The application provides comprehensive metrics for:
- Request/response times
- AI agent performance
- Token usage and costs
- Error rates and types
- User activity patterns

## 🛠️ Development

### Project Structure
```
src/
├── ai_services/          # AI agent system
│   ├── agent_orchestrator.py
│   ├── gemini_agents.py
│   ├── function_tools.py
│   └── workflow_engine.py
├── api/                  # API layer
│   ├── dependencies.py
│   ├── middleware.py
│   └── health.py
├── auth/                 # Authentication
│   ├── firebase_auth.py
│   ├── dependencies.py
│   └── routes.py
├── core/                 # Core utilities
│   ├── config.py
│   ├── logging.py
│   └── exceptions.py
├── database/             # Database layer
│   └── firestore_client.py
├── maps_services/        # Maps integration
│   ├── places_service.py
│   ├── directions_service.py
│   └── geocoding_service.py
├── trip_planner/         # Trip planning logic
│   ├── routes.py
│   ├── services.py
│   └── schemas.py
└── main.py              # Application entry point
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/
```

### Adding New Agents
```python
from src.ai_services.gemini_agents import AgentFactory, AgentRole

# Create specialized agent
agent = AgentFactory.create_custom_agent(
    role=AgentRole.WEATHER_ANALYST,
    capabilities=["weather_forecasting", "climate_analysis"],
    session_id="user_session_123"
)

# Register with orchestrator
orchestrator = get_trip_planner_orchestrator()
await orchestrator.register_agent(agent)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Write comprehensive tests
- Update documentation for new features
- Use type hints throughout
- Add logging for debugging

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Cloud Vertex AI** for advanced AI capabilities
- **Firebase** for authentication and database services
- **Google Maps Platform** for location services
- **FastAPI** for the excellent web framework
- **Pydantic** for robust data validation

## 📞 Support

- **Documentation**: [Full Documentation](docs/)
- **API Reference**: [API Docs](docs/api.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/genai-trip-planner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/genai-trip-planner/discussions)

---

**Made with ❤️ for intelligent travel planning**
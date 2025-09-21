# Development Guide

> Comprehensive development guidelines for the AI-Powered Trip Planner Backend

## Table of Contents

- [Local Development Setup](#local-development-setup)
- [Code Organization](#code-organization)
- [Development Workflow](#development-workflow)
- [Testing Guidelines](#testing-guidelines)
- [Code Quality Standards](#code-quality-standards)
- [Contributing Guidelines](#contributing-guidelines)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)
- [Performance Guidelines](#performance-guidelines)
- [Security Guidelines](#security-guidelines)

---

## Local Development Setup

### Prerequisites

- **Python 3.12+**: Latest Python version with modern features.
- **Poetry**: For dependency management.
- **Docker & Docker Compose**: For containerized development.
- **Google Cloud SDK**: For interacting with GCP services.
- **Firebase CLI**: For managing Firebase projects.
- **Git**: For version control.

### Step-by-Step Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd genai-hackathon-backend
    ```

2.  **Set up Python Environment**
    ```bash
    # Install dependencies
    poetry install

    # Activate virtual environment
    poetry shell
    ```

3.  **Configure Environment Variables**
    ```bash
    # Copy the example environment file
    cp .env.example .env.development

    # Edit .env.development with your local credentials
    # - Firebase project details
    # - Google Cloud project details
    # - Google Maps API key
    # - A strong, random JWT_SECRET_KEY
    ```

4.  **Set up Firebase Emulator**
    ```bash
    # Install Firebase CLI if not already installed
    npm install -g firebase-tools

    # Initialize Firebase Emulators
    firebase init emulators

    # Start emulators
    firebase emulators:start --only auth,firestore
    ```

5.  **Run the Development Server**
    ```bash
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
    ```

6.  **Access the Application**
    - **API**: `http://localhost:8000`
    - **API Docs (Swagger)**: `http://localhost:8000/docs`
    - **API Docs (ReDoc)**: `http://localhost:8000/redoc`
    - **Firebase Emulator UI**: `http://localhost:4000`

### Dockerized Development

For a consistent development environment, use Docker Compose:

```bash
# Start all services (API, database, etc.)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Code Organization

The project follows a modular, feature-based structure within the `src/` directory.

```
src/
├── ai_services/          # AI agent system and orchestration
│   ├── agent_orchestrator.py
│   ├── gemini_agents.py
│   ├── function_tools.py
│   └── workflow_engine.py
├── api/                  # API layer configurations
│   ├── dependencies.py
│   ├── middleware.py
│   └── health.py
├── auth/                 # Authentication and user management
│   ├── firebase_auth.py
│   ├── dependencies.py
│   └── routes.py
├── core/                 # Core utilities and shared components
│   ├── config.py
│   ├── logging.py
│   └── exceptions.py
├── database/             # Database layer and models
│   └── firestore_client.py
├── maps_services/        # Google Maps integration
│   ├── places_service.py
│   ├── directions_service.py
│   └── geocoding_service.py
├── trip_planner/         # Core trip planning business logic
│   ├── routes.py
│   ├── services.py
│   └── schemas.py
└── main.py              # Application entry point
```

### Key Principles

- **Separation of Concerns**: Each module has a distinct responsibility.
- **Dependency Injection**: Services and clients are injected where needed.
- **Feature-Based Modules**: Code is organized by feature (e.g., `trip_planner`, `auth`).
- **Clear Naming Conventions**: Files and directories are named descriptively.

---

## Development Workflow

### Feature Development

1.  **Create a New Branch**
    ```bash
    git checkout -b feature/your-feature-name
    ```

2.  **Implement Changes**
    - Add new features or fix bugs.
    - Follow code quality standards.
    - Add or update tests for your changes.

3.  **Run Quality Checks**
    ```bash
    # Run all checks
    poetry run pre-commit run --all-files

    # Or run individually
    poetry run black .
    poetry run ruff .
    poetry run mypy .
    ```

4.  **Run Tests**
    ```bash
    poetry run pytest
    ```

5.  **Commit and Push**
    ```bash
    git commit -m "feat: Implement amazing feature"
    git push origin feature/your-feature-name
    ```

6.  **Create a Pull Request**
    - Open a PR against the `main` branch.
    - Provide a clear description of your changes.
    - Link to any relevant issues.

### Adding a New API Endpoint

1.  **Define Schema**: Create request/response models in the relevant `schemas.py`.
2.  **Create Service Logic**: Implement business logic in the relevant `services.py`.
3.  **Add Route**: Define the new endpoint in the relevant `routes.py`.
4.  **Add Tests**: Write unit and integration tests for the new endpoint.
5.  **Update Documentation**: Regenerate or update API documentation.

### Adding a New AI Agent

1.  **Define Agent Role**: Add a new role to `AgentRole` enum in `src/ai_services/gemini_agents.py`.
2.  **Create Agent Factory Method**: Add a new creation method to `AgentFactory`.
3.  **Define Capabilities**: Register agent capabilities in `AgentCapabilityRegistry`.
4.  **Create Prompt Template**: Add a new prompt template in `src/ai_services/prompt_templates.py`.
5.  **Integrate into Workflows**: Add the new agent to relevant workflows in `src/ai_services/agent_orchestrator.py`.

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run tests for a specific file
poetry run pytest src/trip_planner/tests/test_services.py

# Run tests with coverage report
poetry run pytest --cov=src
```

### Test Structure

Tests are located alongside the source code in `tests/` subdirectories.

```
src/
└── trip_planner/
    ├── services.py
    └── tests/
        ├── test_services.py
        └── test_routes.py
```

### Types of Tests

- **Unit Tests**: Test individual functions and classes in isolation.
- **Integration Tests**: Test interactions between components (e.g., service and database).
- **End-to-End (E2E) Tests**: Test the entire application flow through API endpoints.

### Writing Tests

- Use `pytest` fixtures for setup and teardown.
- Mock external services using `unittest.mock`.
- Use `httpx.AsyncClient` for testing API endpoints.
- Aim for high test coverage for new code.

### Example Test

```python
# src/trip_planner/tests/test_services.py
import pytest
from unittest.mock import AsyncMock

from src.trip_planner.services import TripPlannerService

@pytest.mark.asyncio
async def test_create_trip_plan():
    # Arrange
    mock_orchestrator = AsyncMock()
    mock_orchestrator.execute_workflow.return_value = {"itinerary": "..."},
    
    service = TripPlannerService(orchestrator_factory=lambda: mock_orchestrator)
    
    # Act
    result = await service.create_trip_plan(
        user_id="test_user",
        trip_request={...}
    )
    
    # Assert
    assert result is not None
    assert "itinerary" in result
    mock_orchestrator.execute_workflow.assert_called_once()
```

---

## Code Quality Standards

### Linting and Formatting

We use `black` for formatting and `ruff` for linting.

```bash
# Format code
poetry run black .

# Lint code
poetry run ruff .
```

### Type Checking

We use `mypy` for static type checking.

```bash
# Run type checking
poetry run mypy .
```

### Pre-commit Hooks

Pre-commit hooks are configured to run these checks automatically before each commit.

```bash
# Install pre-commit hooks
poetry run pre-commit install
```

---

## Contributing Guidelines

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Build process or tooling changes

### Pull Requests

- **Title**: Clear and descriptive title.
- **Description**: Explain the "what" and "why" of your changes.
- **Link Issues**: Reference any related issues.
- **Tests**: Ensure all tests are passing.
- **Review**: Request a review from at least one other developer.

---

## Debugging and Troubleshooting

### Logging

The application uses `structlog` for structured logging.

- **Log Level**: Controlled by `LOG_LEVEL` environment variable (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- **Correlation ID**: `X-Correlation-ID` header is used to trace requests through the system.

### Debugging Endpoints

When `DEBUG=true`, the following endpoints are available:

- `/debug/config`: Display current application configuration.
- `/debug/middleware`: Display registered middleware.
- `/debug/health`: Extended health check with detailed service info.

### Common Issues

- **Firebase Auth Errors**: Ensure your service account key is correct and has the right permissions.
- **Vertex AI Errors**: Check that the Vertex AI API is enabled and your project has a sufficient quota.
- **Dependency Issues**: Run `poetry install` to ensure all dependencies are up to date.

---

## Performance Guidelines

- **Use Async Everywhere**: All I/O-bound operations (database, external APIs) should be `async`.
- **Optimize Database Queries**: Use indexes and efficient query patterns.
- **Implement Caching**: Cache expensive operations and frequently accessed data.
- **Profile Your Code**: Use tools like `cProfile` or `py-spy` to identify performance bottlenecks.

---

## Security Guidelines

- **Never Commit Secrets**: Use environment variables or a secret manager for sensitive data.
- **Validate All Inputs**: Use Pydantic models to validate all incoming data.
- **Sanitize Outputs**: Ensure no sensitive data is leaked in API responses or logs.
- **Use Dependency Injection**: Avoid global state and use FastAPI's dependency injection system.
- **Follow Principle of Least Privilege**: Service accounts should have only the permissions they need.
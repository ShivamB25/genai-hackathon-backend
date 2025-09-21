# Deployment Guide

> Complete deployment guide for the AI-Powered Trip Planner Backend in production environments

## Table of Contents

- [Production Deployment Checklist](#production-deployment-checklist)
- [Environment Configuration](#environment-configuration)
- [Google Cloud Setup](#google-cloud-setup)
- [Firebase Configuration](#firebase-configuration)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment Options](#cloud-deployment-options)
- [Performance Optimization](#performance-optimization)
- [Monitoring & Logging](#monitoring--logging)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

---

## Production Deployment Checklist

### Pre-Deployment Requirements

- [ ] Google Cloud Project with billing enabled
- [ ] Firebase project configured
- [ ] Domain name and SSL certificate
- [ ] Production database (Firestore) setup
- [ ] Environment variables configured
- [ ] Docker image built and tested
- [ ] Load balancer configured (if needed)
- [ ] Monitoring and logging setup
- [ ] Backup and recovery plan

### API Services Setup

- [ ] **Vertex AI API** enabled
- [ ] **Firebase Admin SDK API** enabled
- [ ] **Google Maps Platform APIs** enabled:
  - Places API
  - Directions API
  - Geocoding API
  - Distance Matrix API
  - Maps JavaScript API (for frontend)

### Security Configuration

- [ ] Service account with minimal required permissions
- [ ] API keys restricted to specific APIs and domains
- [ ] Firebase security rules configured
- [ ] CORS origins properly set
- [ ] Rate limiting configured
- [ ] JWT secret key generated (strong, unique)

### Performance & Scaling

- [ ] Resource limits configured
- [ ] Caching strategy implemented
- [ ] Database indexes optimized
- [ ] CDN configured for static assets
- [ ] Health checks configured
- [ ] Auto-scaling rules defined

---

## Environment Configuration

### Production Environment Variables

Create a `.env.production` file with the following variables:

```bash
# =============================================================================
# PRODUCTION ENVIRONMENT CONFIGURATION
# =============================================================================

# Application Settings
APP_NAME=genai-trip-planner
APP_VERSION=0.1.0
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# =============================================================================
# GOOGLE CLOUD CONFIGURATION
# =============================================================================

# Google Cloud Project
GOOGLE_CLOUD_PROJECT=your-production-project
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_CLOUD_ZONE=us-central1-a

# Service Account (use JSON string for containerized deployments)
GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json
# Alternative for containerized environments:
# GOOGLE_CREDENTIALS_JSON='{"type": "service_account", "project_id": "...", ...}'

# =============================================================================
# VERTEX AI CONFIGURATION
# =============================================================================

VERTEX_AI_PROJECT_ID=your-production-project
VERTEX_AI_REGION=us-central1

# Gemini Model Settings
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=8192
GEMINI_TOP_P=0.95
GEMINI_TOP_K=40

# =============================================================================
# FIREBASE CONFIGURATION
# =============================================================================

# Firebase Project
FIREBASE_PROJECT_ID=your-production-project
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your-client-id
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token
FIREBASE_CLIENT_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com

# Firebase Web Configuration
FIREBASE_WEB_API_KEY=your-firebase-web-api-key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_STORAGE_BUCKET=your-project.appspot.com

# Firestore Settings
FIRESTORE_DATABASE_ID=(default)

# =============================================================================
# GOOGLE MAPS API CONFIGURATION
# =============================================================================

GOOGLE_MAPS_API_KEY=your-maps-api-key

# API Service Enablement
MAPS_PLACES_API_ENABLED=true
MAPS_DIRECTIONS_API_ENABLED=true
MAPS_DISTANCE_MATRIX_API_ENABLED=true
MAPS_GEOCODING_API_ENABLED=true
MAPS_TIMEZONE_API_ENABLED=true

# Performance Settings
MAPS_API_RATE_LIMIT=1000
MAPS_REQUESTS_PER_SECOND=50.0
MAPS_BURST_LIMIT=100
MAPS_REQUEST_TIMEOUT=30
MAPS_MAX_RETRIES=3

# Caching Configuration
MAPS_ENABLE_CACHING=true
MAPS_CACHE_TTL=3600
MAPS_CACHE_MAX_SIZE=10000

# =============================================================================
# AI SERVICE CONFIGURATION
# =============================================================================

AI_MODEL_PROVIDER=vertex-ai
AI_FALLBACK_PROVIDER=google-ai
ENABLE_FUNCTION_CALLING=true
MAX_FUNCTION_CALLS=5
FUNCTION_CALL_TIMEOUT=60

# Advanced AI Settings
VERTEX_AI_MAX_RETRIES=3
VERTEX_AI_RETRY_DELAY=2.0
VERTEX_AI_REQUEST_TIMEOUT=120
VERTEX_AI_CONCURRENT_REQUESTS=20

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# JWT Settings (generate strong random key)
JWT_SECRET_KEY=your-super-secure-random-jwt-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_TIME=3600
JWT_REFRESH_EXPIRATION_TIME=604800

# CORS Configuration
ENABLE_CORS=true
ALLOWED_ORIGINS=["https://yourdomain.com", "https://app.yourdomain.com"]
ALLOWED_METHODS=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
ALLOWED_HEADERS=["*"]

# API Security
API_RATE_LIMIT=1000
ENABLE_REQUEST_ID_TRACKING=true

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_TYPE=firestore
DATABASE_TIMEOUT=60
DATABASE_RETRY_ATTEMPTS=3

# Caching
CACHE_TYPE=memory
CACHE_TTL=3600
CACHE_MAX_SIZE=10000

# =============================================================================
# MONITORING AND LOGGING
# =============================================================================

# Logging
ENABLE_CLOUD_LOGGING=true
LOG_FORMAT=structured
LOG_CORRELATION_ID=true

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_PATH=/health
ENABLE_ERROR_TRACKING=true
ERROR_TRACKING_SERVICE=cloud-logging
ENABLE_PERFORMANCE_MONITORING=true
TRACE_SAMPLE_RATE=0.1

# =============================================================================
# BUSINESS LOGIC CONFIGURATION
# =============================================================================

# Trip Planning
DEFAULT_TRIP_DURATION_DAYS=7
MAX_TRIP_DURATION_DAYS=30
MIN_TRIP_DURATION_DAYS=1
DEFAULT_BUDGET_CURRENCY=USD
DEFAULT_SEARCH_RADIUS=50
MAX_PLACES_PER_DAY=8
DEFAULT_COUNTRY=United States
SUPPORTED_LANGUAGES=["en"]
DEFAULT_TIMEZONE=UTC

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Session Management
SESSION_MAX_DURATION=86400
SESSION_IDLE_TIMEOUT=3600
SESSION_CLEANUP_INTERVAL=300
MAX_SESSIONS_PER_USER=3

# Agent Configuration
MAX_ACTIVE_AGENTS=100
AGENT_IDLE_TIMEOUT=1800
AGENT_MAX_ITERATIONS=20
AGENT_RESPONSE_TIMEOUT=180

# =============================================================================
# PRODUCTION SETTINGS
# =============================================================================

# Production Flags
LOCAL_DEVELOPMENT=false
MOCK_EXTERNAL_APIS=false
USE_LOCAL_CREDENTIALS=false
ENABLE_DEBUG_TOOLBAR=false
ENABLE_PROFILER=false
ENABLE_HOT_RELOAD=false
ENABLE_TEST_MODE=false
```

### Environment Variable Security

**Never commit production `.env` files to version control!**

Use one of these secure methods:

#### Method 1: Secret Management Service

```bash
# Google Cloud Secret Manager
gcloud secrets create app-env-vars --data-file=.env.production

# AWS Secrets Manager
aws secretsmanager create-secret \
  --name "trip-planner/env" \
  --secret-string file://.env.production

# Azure Key Vault
az keyvault secret set \
  --vault-name "TripPlannerVault" \
  --name "env-vars" \
  --file .env.production
```

#### Method 2: Container Environment Variables

```yaml
# Docker Compose
version: '3.8'
services:
  trip-planner:
    image: trip-planner:latest
    environment:
      - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
      - FIREBASE_PROJECT_ID=${FIREBASE_PROJECT_ID}
      - GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY}
    env_file:
      - .env.production
```

#### Method 3: Kubernetes ConfigMap/Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trip-planner-config
data:
  APP_NAME: "genai-trip-planner"
  ENVIRONMENT: "production"
  DEBUG: "false"
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: trip-planner-secret
type: Opaque
data:
  GOOGLE_MAPS_API_KEY: <base64-encoded-key>
  JWT_SECRET_KEY: <base64-encoded-key>
```

---

## Google Cloud Setup

### 1. Create and Configure Project

```bash
# Create new project
gcloud projects create your-trip-planner-prod --name="Trip Planner Production"

# Set current project
gcloud config set project your-trip-planner-prod

# Enable billing (required for APIs)
gcloud billing projects link your-trip-planner-prod --billing-account=BILLING_ACCOUNT_ID
```

### 2. Enable Required APIs

```bash
# Enable all required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  firebase.googleapis.com \
  firestore.googleapis.com \
  places-backend.googleapis.com \
  directions-backend.googleapis.com \
  geocoding-backend.googleapis.com \
  maps-backend.googleapis.com \
  cloudrun.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

### 3. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create trip-planner-prod \
  --display-name="Trip Planner Production Service Account" \
  --description="Service account for Trip Planner production deployment"

# Get service account email
SA_EMAIL=$(gcloud iam service-accounts list \
  --filter="displayName:Trip Planner Production Service Account" \
  --format="value(email)")

# Assign required roles
gcloud projects add-iam-policy-binding your-trip-planner-prod \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding your-trip-planner-prod \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/firebase.admin"

gcloud projects add-iam-policy-binding your-trip-planner-prod \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/datastore.user"

gcloud projects add-iam-policy-binding your-trip-planner-prod \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding your-trip-planner-prod \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/monitoring.metricWriter"

# Create and download service account key
gcloud iam service-accounts keys create service-account-prod.json \
  --iam-account="${SA_EMAIL}"
```

### 4. Configure Vertex AI

```bash
# Set up Vertex AI location
gcloud ai-platform models list --region=us-central1

# Test Vertex AI access
gcloud ai-platform jobs submit training dummy-job \
  --region=us-central1 \
  --dry-run
```

### 5. Configure Google Maps API

```bash
# Create API key for Maps services
gcloud alpha services api-keys create \
  --display-name="Trip Planner Maps API Key"

# Get the API key
API_KEY=$(gcloud alpha services api-keys list \
  --filter="displayName:Trip Planner Maps API Key" \
  --format="value(name)")

# Restrict API key to specific APIs
gcloud alpha services api-keys update $API_KEY \
  --api-target="places-backend.googleapis.com" \
  --api-target="directions-backend.googleapis.com" \
  --api-target="geocoding-backend.googleapis.com" \
  --api-target="maps-backend.googleapis.com"

# Restrict to specific domains (replace with your domains)
gcloud alpha services api-keys update $API_KEY \
  --allowed-referrers="https://yourdomain.com/*" \
  --allowed-referrers="https://app.yourdomain.com/*"
```

---

## Firebase Configuration

### 1. Create Firebase Project

```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login to Firebase
firebase login

# Create new project (or use existing Google Cloud project)
firebase projects:create your-trip-planner-prod --display-name="Trip Planner Production"

# Set current project
firebase use your-trip-planner-prod
```

### 2. Initialize Firebase Services

```bash
# Initialize Firebase in project directory
firebase init

# Select services:
# - Authentication
# - Firestore
# - Functions (optional)
# - Hosting (if needed)
```

### 3. Configure Firestore Security Rules

Create `firestore.rules`:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Users can only access their own trips
    match /trips/{tripId} {
      allow read, write: if request.auth != null && 
        (request.auth.uid == resource.data.user_id || 
         request.auth.uid in resource.data.shared_with);
    }
    
    // Users can only access their own AI sessions
    match /ai_sessions/{sessionId} {
      allow read, write: if request.auth != null && 
        request.auth.uid == resource.data.user_id;
    }
    
    // Public read access for places data (cached)
    match /places_cache/{placeId} {
      allow read: if true;
      allow write: if false; // Only server can write
    }
  }
}
```

### 4. Configure Firebase Authentication

```bash
# Enable authentication providers
firebase auth:import users.json --hash-algo=PBKDF2_SHA256

# Configure OAuth providers (via Firebase Console)
# - Google
# - Facebook  
# - Apple
# - Email/Password
```

### 5. Set up Firebase Admin SDK

```javascript
// firebase-admin-init.js
const admin = require('firebase-admin');
const serviceAccount = require('./service-account-prod.json');

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: 'https://your-trip-planner-prod-default-rtdb.firebaseio.com'
});

module.exports = admin;
```

### 6. Deploy Security Rules

```bash
# Deploy Firestore rules
firebase deploy --only firestore:rules

# Deploy authentication configuration
firebase deploy --only auth
```

---

## Docker Deployment

### 1. Production Dockerfile

Create an optimized production `Dockerfile`:

```dockerfile
# Multi-stage build for production
FROM python:3.12-slim AS builder

# Set build arguments
ARG POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set environment variables
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/opt/poetry/cache \
    POETRY_HOME="/opt/poetry"

# Copy dependency files
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.12-slim AS production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Copy service account (if using file-based auth)
COPY --chown=appuser:appuser service-account-prod.json ./service-account.json

# Create necessary directories
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Docker Compose for Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  trip-planner:
    build:
      context: .
      dockerfile: Dockerfile
    image: trip-planner:latest
    container_name: trip-planner-prod
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env.production
    volumes:
      - ./logs:/app/logs
      - ./service-account-prod.json:/app/service-account.json:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.trip-planner.rule=Host(`api.yourdomain.com`)"
      - "traefik.http.routers.trip-planner.tls=true"
      - "traefik.http.routers.trip-planner.tls.certresolver=letsencrypt"
    
  # Optional: Reverse proxy with SSL
  traefik:
    image: traefik:v2.10
    container_name: traefik
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik.yml:/etc/traefik/traefik.yml:ro
      - ./traefik-dynamic.yml:/etc/traefik/dynamic.yml:ro
      - traefik-acme:/acme
    labels:
      - "traefik.enable=true"

volumes:
  traefik-acme:
```

### 3. Build and Deploy

```bash
# Build production image
docker build -t trip-planner:latest .

# Tag for registry
docker tag trip-planner:latest gcr.io/your-trip-planner-prod/trip-planner:latest

# Push to Google Container Registry
docker push gcr.io/your-trip-planner-prod/trip-planner:latest

# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Docker Registry Options

#### Google Container Registry

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/your-trip-planner-prod/trip-planner:v1.0.0 .
docker push gcr.io/your-trip-planner-prod/trip-planner:v1.0.0
```

#### Docker Hub

```bash
# Login to Docker Hub
docker login

# Build and push
docker build -t yourusername/trip-planner:v1.0.0 .
docker push yourusername/trip-planner:v1.0.0
```

#### AWS ECR

```bash
# Get login token
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

# Build and push
docker build -t trip-planner .
docker tag trip-planner:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/trip-planner:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/trip-planner:latest
```

---

## Cloud Deployment Options

### 1. Google Cloud Run

#### Deploy to Cloud Run

```bash
# Deploy directly from source
gcloud run deploy trip-planner \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ENVIRONMENT=production \
  --set-secrets="/app/service-account.json=projects/your-trip-planner-prod/secrets/service-account-key:latest" \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --concurrency 100 \
  --timeout 300

# Or deploy from container registry
gcloud run deploy trip-planner \
  --image gcr.io/your-trip-planner-prod/trip-planner:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```

#### Cloud Run Configuration YAML

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: trip-planner
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/your-trip-planner-prod/trip-planner:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: production
        - name: GOOGLE_CLOUD_PROJECT
          value: your-trip-planner-prod
        volumeMounts:
        - name: service-account
          mountPath: /app/service-account.json
          subPath: service-account.json
      volumes:
      - name: service-account
        secret:
          secretName: service-account-key
```

#### Deploy with YAML

```bash
gcloud run services replace service.yaml --region us-central1
```

### 2. Google Kubernetes Engine (GKE)

#### Create GKE Cluster

```bash
# Create cluster
gcloud container clusters create trip-planner-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type e2-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials trip-planner-cluster --zone us-central1-a
```

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trip-planner
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trip-planner
  template:
    metadata:
      labels:
        app: trip-planner
    spec:
      containers:
      - name: trip-planner
        image: gcr.io/your-trip-planner-prod/trip-planner:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: production
        envFrom:
        - configMapRef:
            name: trip-planner-config
        - secretRef:
            name: trip-planner-secret
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: trip-planner-service
spec:
  selector:
    app: trip-planner
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trip-planner-ingress
  annotations:
    kubernetes.io/ingress.global-static-ip-name: trip-planner-ip
    networking.gke.io/managed-certificates: trip-planner-ssl-cert
spec:
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: trip-planner-service
            port:
              number: 80
```

#### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s-configmap.yaml
kubectl apply -f k8s-secret.yaml
kubectl apply -f k8s-deployment.yaml

# Check deployment
kubectl get pods
kubectl get services
kubectl get ingress
```

### 3. AWS ECS/Fargate

#### Task Definition

```json
{
  "family": "trip-planner",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "trip-planner",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/trip-planner:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "GOOGLE_MAPS_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:123456789012:secret:trip-planner/maps-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/trip-planner",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Deploy to ECS

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster trip-planner-cluster \
  --service-name trip-planner-service \
  --task-definition trip-planner:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678,subnet-87654321],securityGroups=[sg-12345678],assignPublicIp=ENABLED}"
```

### 4. Azure Container Instances

```bash
# Create resource group
az group create --name TripPlannerRG --location eastus

# Create container instance
az container create \
  --resource-group TripPlannerRG \
  --name trip-planner \
  --image yourdockerhub/trip-planner:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production \
  --secure-environment-variables GOOGLE_MAPS_API_KEY=your-api-key \
  --dns-name-label trip-planner-api
```

---

## Performance Optimization

### 1. Application Performance

#### FastAPI Optimization

```python
# main.py optimizations
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()

# Add compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)

# Optimize startup
@app.on_event("startup")
async def startup_event():
    # Pre-warm connections
    await initialize_services()
```

#### Database Optimization

```python
# Firestore optimization
from google.cloud import firestore

# Use connection pooling
client = firestore.AsyncClient()

# Add indexes for frequent queries
# In Firestore console, add composite indexes for:
# - users: (uid, created_at)
# - trips: (user_id, status, created_at)
# - trips: (destination, start_date)
```

#### Caching Strategy

```python
# Redis caching for production
import redis.asyncio as redis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# Initialize Redis cache
@app.on_event("startup")
async def startup():
    redis_client = redis.from_url("redis://localhost:6379")
    FastAPICache.init(RedisBackend(redis_client), prefix="trip-planner")

# Cache expensive operations
from fastapi_cache.decorator import cache

@cache(expire=3600)  # 1 hour cache
async def get_places_data(location: str):
    # Expensive Maps API call
    return await fetch_places(location)
```

### 2. Infrastructure Scaling

#### Auto Scaling Configuration

```yaml
# Cloud Run autoscaling
annotations:
  autoscaling.knative.dev/minScale: "2"
  autoscaling.knative.dev/maxScale: "100"
  run.googleapis.com/cpu-throttling: "false"
  run.googleapis.com/memory: "4Gi"
  run.googleapis.com/cpu: "2"

# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trip-planner-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trip-planner
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Load Balancing

```nginx
# nginx.conf for load balancing
upstream trip_planner {
    server backend1.yourdomain.com:8000 weight=3;
    server backend2.yourdomain.com:8000 weight=3;
    server backend3.yourdomain.com:8000 weight=2;
    keepalive 32;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    
    location / {
        proxy_pass http://trip_planner;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
}
```

### 3. CDN Configuration

#### Cloudflare Setup

```javascript
// cloudflare-workers.js
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  // Cache static content
  if (request.url.includes('/docs') || request.url.includes('/openapi.json')) {
    const cacheKey = new Request(request.url, request)
    const cache = caches.default
    
    let response = await cache.match(cacheKey)
    if (!response) {
      response = await fetch(request)
      const headers = new Headers(response.headers)
      headers.set('Cache-Control', 'public, max-age=86400')
      
      response = new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: headers
      })
      
      event.waitUntil(cache.put(cacheKey, response.clone()))
    }
    return response
  }
  
  return fetch(request)
}
```

---

## Monitoring & Logging

### 1. Google Cloud Monitoring

#### Set up monitoring

```python
# monitoring.py
from google.cloud import monitoring_v3
from google.cloud import error_reporting

# Initialize clients
monitoring_client = monitoring_v3.MetricServiceClient()
error_client = error_reporting.Client()

# Custom metrics
def record_trip_generation_time(duration: float):
    series = monitoring_v3.TimeSeries()
    series.metric.type = "custom.googleapis.com/trip_generation_duration"
    series.resource.type = "global"
    
    point = monitoring_v3.Point()
    point.value.double_value = duration
    point.interval.end_time.GetCurrentTime()
    series.points = [point]
    
    monitoring_client.create_time_series(
        name=f"projects/{PROJECT_ID}",
        time_series=[series]
    )

# Error reporting
def report_error(error: Exception):
    error_client.report_exception()
```

#### Alert Policies

```yaml
# alerting.yaml
displayName: "Trip Planner High Error Rate"
conditions:
  - displayName: "Error rate > 5%"
    conditionThreshold:
      filter: 'resource.type="cloud_run_revision" AND resource.labels.service_name="trip-planner"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 0.05
      duration: 300s
notificationChannels:
  - "projects/your-project/notificationChannels/email-channel"
```

### 2. Structured Logging

```python
# logging_config.py
import structlog
from google.cloud import logging as cloud_logging

# Initialize Google Cloud Logging
cloud_logging.Client().setup_logging()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    context_class=dict,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info(
    "Trip generation completed",
    user_id="user123",
    trip_id="trip456",
    duration=45.2,
    tokens_used=1500
)
```

### 3. Application Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
from fastapi import FastAPI, Request
import time

# Define metrics
REQUEST_COUNT = Counter(
    'trip_planner_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'trip_planner_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

ACTIVE_SESSIONS = Gauge(
    'trip_planner_active_sessions',
    'Number of active user sessions'
)

TRIP_GENERATION_TIME = Histogram(
    'trip_planner_generation_duration_seconds',
    'Time to generate a trip plan'
)

# Middleware to collect metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response
```

---

## Security Considerations

### 1. API Security

#### Rate Limiting

```python
# rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.state.limiter = limiter
@app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply to endpoints
@app.post("/api/v1/trips/plan")
@limiter.limit("5/minute")  # 5 trip generations per minute
async def create_trip_plan(request: Request):
    pass
```

#### Input Validation

```python
# security.py
from pydantic import BaseModel, validator
from typing import List
import re

class SecureTripRequest(BaseModel):
    destination: str
    traveler_count: int
    
    @validator('destination')
    def validate_destination(cls, v):
        # Sanitize input
        if not re.match(r'^[a-zA-Z0-9\s\-,\.]+$', v):
            raise ValueError('Invalid destination format')
        return v.strip()
    
    @validator('traveler_count')
    def validate_traveler_count(cls, v):
        if v < 1 or v > 50:
            raise ValueError('Traveler count must be between 1 and 50')
        return v
```

#### HTTPS and CORS

```python
# security_middleware.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Force HTTPS in production
if settings.environment == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)
```

### 2. Secrets Management

#### Google Secret Manager

```python
# secrets.py
from google.cloud import secretmanager

class SecretManager:
    def __init__(self, project_id: str):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id
    
    def get_secret(self, secret_id: str, version: str = "latest") -> str:
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")

# Usage
secret_manager = SecretManager("your-project-id")
api_key = secret_manager.get_secret("google-maps-api-key")
```

### 3. Container Security

#### Secure Dockerfile

```dockerfile
# Use official Python runtime as base image
FROM python:3.12-slim

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Switch to non-root user
USER appuser

# Set work directory
WORKDIR /app

# Copy and install dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=appuser:appuser . .

# Run security scan (optional)
# RUN pip install safety && safety check

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Troubleshooting

### Common Issues

#### 1. Firebase Authentication Errors

**Error**: `Invalid authentication credentials`

**Solution**:
```bash
# Verify service account key
gcloud auth activate-service-account --key-file=service-account.json

# Check Firebase project ID
firebase projects:list

# Verify environment variables
echo $FIREBASE_PROJECT_ID
echo $GOOGLE_APPLICATION_CREDENTIALS
```

#### 2. Vertex AI Connection Issues

**Error**: `Failed to connect to Vertex AI`

**Solution**:
```bash
# Check API enablement
gcloud services list --enabled --filter="aiplatform"

# Test authentication
gcloud ai-platform models list --region=us-central1

# Verify service account permissions
gcloud projects get-iam-policy your-project-id \
  --flatten="bindings[].members" \
  --format="table(bindings.role)" \
  --filter="bindings.members:your-service-account@your-project.iam.gserviceaccount.com"
```

#### 3. Google Maps API Quota Exceeded

**Error**: `You have exceeded your request quota`

**Solution**:
```bash
# Check quota usage
gcloud logging read "protoPayload.serviceName=maps-backend.googleapis.com" \
  --limit=50 --format="table(timestamp, protoPayload.response.error.message)"

# Increase quota (if needed)
# Go to Google Cloud Console > APIs & Services > Quotas
# Select Maps APIs and request quota increase

# Implement caching
# See performance optimization section
```

#### 4. Database Connection Issues

**Error**: `Firestore timeout`

**Solution**:
```python
# Increase timeout in configuration
DATABASE_TIMEOUT=120

# Add retry logic
from google.api_core import retry

@retry.Retry(predicate=retry.if_exception_type(Exception))
async def write_to_firestore(data):
    await firestore_client.collection('trips').add(data)
```

#### 5. High Memory Usage

**Error**: Container killed due to OOM

**Solution**:
```python
# Monitor memory usage
import psutil
import gc

def check_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    if memory_mb > 1500:  # 1.5GB threshold
        gc.collect()
    
# Optimize large data processing
async def process_large_dataset(data):
    # Process in chunks
    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        await process_chunk(chunk)
        gc.collect()  # Force garbage collection
```

### Debugging Commands

#### Container Debugging

```bash
# Check container logs
docker logs trip-planner-prod

# Execute into running container
docker exec -it trip-planner-prod /bin/bash

# Check resource usage
docker stats trip-planner-prod

# Inspect container configuration
docker inspect trip-planner-prod
```

#### Kubernetes Debugging

```bash
# Check pod status
kubectl get pods -l app=trip-planner

# View pod logs
kubectl logs -l app=trip-planner --tail=100

# Describe pod for events
kubectl describe pod <pod-name>

# Execute into pod
kubectl exec -it <pod-name> -- /bin/bash

# Check resource usage
kubectl top pods -l app=trip-planner
```

#### Cloud Run Debugging

```bash
# Check service status
gcloud run services describe trip-planner --region=us-central1

# View logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=trip-planner" \
  --limit=50 --format="table(timestamp, textPayload)"

# Check revisions
gcloud run revisions list --service=trip-planner --region=us-central1
```

### Performance Debugging

#### Database Query Optimization

```python
# Enable Firestore query profiling
import logging
logging.basicConfig()
logging.getLogger('google.cloud.firestore_v1').setLevel(logging.DEBUG)

# Analyze slow queries
@app.middleware("http")
async def log_slow_queries(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    if duration > 2.0:  # Log queries taking > 2 seconds
        logger.warning(
            "Slow request detected",
            path=request.url.path,
            method=request.method,
            duration=duration
        )
    
    return response
```

#### Memory Profiling

```python
# memory_profiler.py
from memory_profiler import profile
import tracemalloc

# Start tracing
tracemalloc.start()

@profile
def memory_intensive_function():
    # Your code here
    pass

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

---

## Backup and Recovery

### Database Backup

```bash
# Export Firestore data
gcloud firestore export gs://your-backup-bucket/firestore-backup-$(date +%Y%m%d)

# Schedule regular backups
gcloud scheduler jobs create http firestore-backup \
  --schedule="0 2 * * *" \
  --uri="https://firestore.googleapis.com/v1/projects/your-project/databases/(default):exportDocuments" \
  --http-method=POST \
  --headers="Authorization=Bearer $(gcloud auth print-access-token)" \
  --message-body='{"outputUriPrefix":"gs://your-backup-bucket/scheduled-backup"}'
```

### Application State Backup

```python
# backup_manager.py
import asyncio
from datetime import datetime
from google.cloud import storage

class BackupManager:
    def __init__(self, bucket_name: str):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
    
    async def backup_user_data(self, user_id: str):
        # Export user trips and preferences
        user_data = await get_user_data(user_id)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        blob_name = f"user-backups/{user_id}/{timestamp}.json"
        
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(user_data, indent=2))
        
        return blob_name
```

### Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 4 hours
2. **RPO (Recovery Point Objective)**: 1 hour
3. **Backup Frequency**: Every 6 hours
4. **Geographic Redundancy**: Multi-region deployment

```bash
# Disaster recovery script
#!/bin/bash

# dr-recovery.sh
echo "Starting disaster recovery..."

# 1. Switch to backup region
gcloud config set compute/region us-west1

# 2. Deploy from backup image
gcloud run deploy trip-planner-dr \
  --image gcr.io/your-project/trip-planner:latest \
  --region us-west1 \
  --allow-unauthenticated

# 3. Restore database from backup
gsutil -m cp -r gs://your-backup-bucket/latest-backup gs://temp-restore/
gcloud firestore import gs://temp-restore/latest-backup

# 4. Update DNS to point to DR instance
# (Manual step or automated with Cloud DNS)

echo "Disaster recovery complete"
```

---

This comprehensive deployment guide provides all the necessary information for deploying the AI-Powered Trip Planner Backend in production environments. For specific deployment scenarios or additional questions, refer to the troubleshooting section or create a support ticket.
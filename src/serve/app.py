from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from src.serve.routers import xg_router
from src.serve.loaders import get_xg_model


app = FastAPI(
    title="Football Analytics API",
    description="Expected Goals (xG) prediction API using StatsBomb data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(xg_router.router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to visualization page."""
    return RedirectResponse(url="/visualize")


@app.get("/visualize", include_in_schema=False)
async def visualize():
    """Serve the interactive xG visualization page."""
    # Serve the HTML file from the static directory
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        # Fallback redirect to docs if static file not found
        return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "Football Analytics API",
        "version": "1.0.0",
    }


@app.on_event("startup")
async def startup_event():
    """
    Load models on startup.
    This ensures models are loaded once when the server starts,
    not on every request.
    """
    print("\n" + "=" * 60)
    print("🚀 Starting Football Analytics API")
    print("=" * 60)

    try:
        # Load xG model
        model = get_xg_model()
        print("✅ xG model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Features: distance, angle")

    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        print("   The API will start, but predictions will fail.")
        print("   Please train the model first:")
        print("   → python -m src.tasks.xg.train.train_xg")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

    print("=" * 60)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("⚽ xG Endpoint: POST http://localhost:8000/v1/xg/score")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\n👋 Shutting down Football Analytics API")

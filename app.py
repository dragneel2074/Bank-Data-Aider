from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes.analysis_routes import analysis_router
import os

app = FastAPI(
    title="Banking Data Analysis System",
    description="API for analyzing banking transaction data",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])

# Serve static files
@app.get("/")
async def index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Enable debug mode for development
    debug_mode = os.environ.get('ENVIRONMENT') == 'development'
    print(f"Starting server in {'debug' if debug_mode else 'production'} mode")  # Debug print
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=debug_mode)
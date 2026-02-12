# Entry point: starts the FastAPI server that exposes /health and /evaluate endpoints
import uvicorn
from ai_eval.service.api import app
# Runs on 0.0.0.0:8000 so it's accessible from localhost and network
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Run the API service for AI evaluation
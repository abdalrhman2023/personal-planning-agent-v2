"""Main application entry point for the Personal Planning Agent (Agent 3).

Starts the FastAPI server with all planning routes mounted.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.api.routes import router as planning_router

app = FastAPI(
    title="Personal Planning Agent (Agent 3)",
    description=(
        "Personal Planning Agent of the 5-Agent Generative AI Personal Life & Study Coach. "
        "Provides deterministic schedule optimization, task deconstruction, Google Calendar sync, "
        "and multi-agent telemetry coordination."
    ),
    version="1.0.0",
)

# Enable CORS cleanly without combining wildcard origins with credentials
allow_origins = list(
    getattr(
        config,
        "CORS_ALLOW_ORIGINS",
        ["http://localhost:3000", "http://localhost:8000"],
    )
)
allow_credentials = "*" not in allow_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(planning_router)


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint to verify agent readiness."""
    return {
        "status": "online",
        "agent": "Personal Planning Agent (Agent 3)",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

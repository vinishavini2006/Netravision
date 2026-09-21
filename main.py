"""
NetraVision Cloud Entry Point
Launches FastAPI Uvicorn server respecting the environment PORT.
"""

import os
import uvicorn
from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[*] Starting NetraVision Cloud Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

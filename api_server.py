import sys
import os

# Set root directory to sys.path to allow imports from src
sys.path.append(os.path.join(os.getcwd(), "src"))

# Now we can safely import the backend app
from src.api_server import app

# This allows the server to be run via `python api_server.py` or directly by uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

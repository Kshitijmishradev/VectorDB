"""
Run the vector DB API as an actual local server:

    python3 run_api.py

Then hit it with curl, e.g.:

    curl -X POST localhost:8000/collections/demo -H "Content-Type: application/json" \
         -d '{"dim": 4, "nlist": 2, "pq_m": 2}'

Or open http://localhost:8000/docs for the interactive Swagger UI FastAPI
generates automatically from the request/response schemas in api.py.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("vectordb.api:app", host="0.0.0.0", port=8000, reload=False)

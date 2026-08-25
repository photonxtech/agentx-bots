import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.routes.upload import router as upload_router
from backend.routes.chat import router as chat_router
from backend.routes.sessions import router as sessions_router

app = FastAPI(
    title="WeNext AI PDF Copilot",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# CORS_ALLOWED_ORIGINS in .env: comma-separated list, e.g.
#   CORS_ALLOWED_ORIGINS=https://your-frontend.easypanel.host
# Falls back to localhost dev origins if unset, so this stays a no-op change
# locally but is safe to set strictly once deployed. allow_origins=["*"] was
# a real gap — it let any website's JS call this API directly.
_cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
    or ["http://localhost:8000", "http://127.0.0.1:8000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(sessions_router)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    upload_path = openapi_schema.get("paths", {}).get("/upload", {})
    if upload_path:
        post_op = upload_path.get("post", {})
        request_body = post_op.get("requestBody", {})
        multipart = request_body.get("content", {}).get("multipart/form-data", {})
        schema = multipart.get("schema", {})

        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            schema = openapi_schema.get("components", {}).get("schemas", {}).get(ref_name, {})

        properties = schema.get("properties", {})
        files_prop = properties.get("files")

        if files_prop and files_prop.get("items"):
            files_prop["items"] = {
                "type": "string",
                "format": "binary",
                "contentMediaType": "application/octet-stream"
            }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi



@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("frontend/templates/index.html")
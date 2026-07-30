from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from backend.routes.upload import router as upload_router
from backend.routes.chat import router as chat_router
from backend.routes.sessions import router as sessions_router

app = FastAPI(
    title="WeNext AI PDF Copilot",
    version="1.0.0"
)

# Register API routes
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(sessions_router)

# Serve CSS, JS and Images
app.mount(
    "/static",
    StaticFiles(directory="frontend/static"),
    name="static"
)

# HTML Templates
templates = Jinja2Templates(directory="frontend/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )
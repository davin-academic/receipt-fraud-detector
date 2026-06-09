from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.receipts import router as receipts_router
from app.routes.test_receipts import router as test_receipts_router

UPLOAD_DIR = Path(__file__).parent / "uploads"

@asynccontextmanager
async def lifespan(app: FastAPI):
    (UPLOAD_DIR / "tmp").mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / "saved").mkdir(parents=True, exist_ok=True)
    yield

app = FastAPI(title="Receipt Fraud Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(receipts_router)
app.include_router(test_receipts_router)

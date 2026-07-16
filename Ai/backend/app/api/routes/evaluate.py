from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_admin, get_db
from app.api.schemas import EvalRequest, EvalRunRead
from app.database.models import AdminUser, EvalRun
from app.services.eval_service import run_evaluation

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate/{website_id}", response_model=EvalRunRead)
async def evaluate_website(
    website_id: int,
    payload: EvalRequest,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> EvalRun:
    cases = [case.model_dump() for case in payload.cases]
    return await run_evaluation(db, website_id, cases)


@router.get("/evaluate/{website_id}", response_model=list[EvalRunRead])
def list_eval_runs(
    website_id: int,
    db: Session = Depends(get_db),
    _admin: AdminUser = Depends(get_current_admin),
) -> list[EvalRun]:
    return (
        db.query(EvalRun)
        .filter(EvalRun.website_id == website_id)
        .order_by(EvalRun.started_at.desc())
        .all()
    )

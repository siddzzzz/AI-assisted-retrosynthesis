"""FastAPI Backend Server for the AI-Assisted Retrosynthesis Studio."""

import os
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from retro_engine.pipeline import RetrosynthesisEngine, BENCHMARK_PRESETS
from retro_engine.chem.mol_utils import calculate_mol_properties, render_mol_svg, canonicalize_smiles


app = FastAPI(
    title="AI-Assisted Retrosynthesis & Synthesis Planning Engine",
    description="Next-generation chemoinformatics and multi-objective retrosynthesis engine with physical chemistry and green Pareto optimization.",
    version="1.0.0",
)

# Initialize the Retrosynthesis Engine
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
engine = RetrosynthesisEngine(data_dir=DATA_DIR)

# Mount static web files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ValidateRequest(BaseModel):
    smiles: str


class PlanRequest(BaseModel):
    smiles: str
    max_depth: int = 5
    max_routes: int = 6
    time_limit_sec: float = 10.0


class SopRequest(BaseModel):
    plan: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the chemical studio frontend."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>AI Retrosynthesis Studio is starting...</h1>")


@app.get("/api/presets")
async def get_presets():
    """Return benchmark chemical presets for instant demonstration."""
    return {"presets": BENCHMARK_PRESETS}


@app.post("/api/validate")
async def validate_molecule(req: ValidateRequest):
    """Validate SMILES string and return 2D structure SVG and physicochemical properties."""
    smi = req.smiles.strip()
    if not smi:
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty.")

    props = engine.analyze_target(smi)
    if not props.get("valid"):
        raise HTTPException(status_code=422, detail=props.get("error", "Invalid molecular structure"))

    return props


@app.post("/api/plan-synthesis")
async def plan_synthesis(req: PlanRequest):
    """Execute multi-step retrosynthesis planning and return scored routes."""
    smi = req.smiles.strip()
    if not smi:
        raise HTTPException(status_code=400, detail="Target SMILES cannot be empty.")

    result = engine.plan_synthesis(
        target_smiles=smi,
        max_depth=req.max_depth,
        max_routes=req.max_routes,
        time_limit_sec=req.time_limit_sec,
    )

    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result.get("error", "Failed to plan synthesis"))

    return result


@app.post("/api/export-sop")
async def export_sop(req: SopRequest):
    """Generate laboratory Standard Operating Procedure in Markdown format."""
    sop_markdown = engine.generate_laboratory_sop(req.plan)
    return {"sop_markdown": sop_markdown}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=True)

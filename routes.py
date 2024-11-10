from fastapi import APIRouter, UploadFile, File
from typing import List
from services import analyze_videos

router = APIRouter()

@router.post("/analyze-videos")
async def analyze_videos_endpoint(files: List[UploadFile] = File(...)):
    return await analyze_videos(files)

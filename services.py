import cv2
import torch
import asyncio
import tempfile
import os
from typing import List, Dict
from fastapi import UploadFile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("E:\Company\Projects\yolo11n.pt")

# Line coordinates for vehicle counting
LINE_START = (100, 500)
LINE_END = (1820, 500)

async def analyze_videos(files: List[UploadFile]) -> Dict[str, Dict[int, int]]:
    with ProcessPoolExecutor() as executor:
        tasks = [process_video(file, executor) for file in files]
        results = await asyncio.gather(*tasks)
    
    traffic_summary = {
        "most_traffic": max(results, key=lambda x: sum(x["counts"].values())),
        "least_traffic": min(results, key=lambda x: sum(x["counts"].values())),
        "moderate_traffic": sorted(results, key=lambda x: sum(x["counts"].values()))[1]
    }
    return traffic_summary

async def process_video(file: UploadFile, executor) -> Dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_video_path = temp_file.name
        temp_file.write(await file.read())

    # Run tracking in a separate thread
    vehicle_counts = await asyncio.get_event_loop().run_in_executor(executor, run_tracker, temp_video_path)
    
    # Clean up the temporary video file
    os.remove(temp_video_path)
    
    return {
        "filename": file.filename,
        "counts": vehicle_counts
    }

def run_tracker(video_path: str) -> Dict[int, int]:
    """
    Track vehicles using YOLOv8 model and count them based on class.
    """
    cap = cv2.VideoCapture(video_path)
    vehicle_counts = defaultdict(int)
    track_history = defaultdict(list)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        
        if results[0].boxes:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))

                if len(track) > 2:
                    prev_point = track[-2]
                    curr_point = track[-1]

                    if (prev_point[1] < LINE_START[1] and curr_point[1] >= LINE_START[1]) or \
                       (prev_point[1] > LINE_START[1] and curr_point[1] <= LINE_START[1]):
                        vehicle_counts[int(cls)] += 1
    
    cap.release()
    return vehicle_counts

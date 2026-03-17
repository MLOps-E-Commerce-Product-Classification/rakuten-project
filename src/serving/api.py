from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
import math
import numpy as np
from pathlib import Path
import logging

from src.inference.run_image_inference import run_image_inference
from src.inference.run_text_inference import run_text_inference  

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Base paths (robust für Docker) ----
BASE_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = BASE_DIR / "data/raw/X_test_update.csv"
IMAGE_DIR = BASE_DIR / "data/raw/images/image_test"

# ---- Lazy load dataframe ----
df = None

def get_df():
    global df
    if df is None:
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"{CSV_PATH} not found")
        logger.info(f"Loading CSV from {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, index_col=0)
    return df


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, (int, str)):
        return obj
    else:
        return str(obj)


def combine_probabilities(image_probs: dict, text_probs: dict, top_k: int):
    combined = {}

    all_codes = set(image_probs.keys()).union(text_probs.keys())

    for code in all_codes:
        combined[code] = (image_probs.get(code, 0) + text_probs.get(code, 0)) / 2

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {"rakuten_code": int(code), "probability": float(prob)}
        for code, prob in top
    ]


api = FastAPI(
    title="Rakuten Inference API",
    description="API to generate Rakuten label based on image and/or text using test CSV and images"
)


class InferenceRequest(BaseModel):
    ids: List[int]
    mode: str = "both"       # "text", "image", "both"
    top_k: int = 27


@api.get('/')
async def get_api():
    return {'message': 'Rakuten Inference API is up and running'}


@api.post("/inference")
async def get_label(request: InferenceRequest):
    try:
        df = get_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    results = []

    for id_ in request.ids:
        if id_ < 84916:
            id_ += 84916

        if id_ not in df.index:
            results.append({"id": id_, "error": "ID not found in CSV"})
            continue

        row = df.loc[id_]
        logger.info(f"Processing ID {id_}")

        text_input = {
            "designation": row.get("designation", ""),
            "description": row.get("description", "")
        }

        productid = row["productid"]
        imageid = row["imageid"]
        image_filename = f"image_{imageid}_product_{productid}.jpg"
        image_path = IMAGE_DIR / image_filename

        entry_result = {"id": id_}

        # ---- Text inference ----
        if request.mode in ["text", "both"]:
            text_res = run_text_inference(
                text_input=text_input,
                top_k=request.top_k
            )
            entry_result["text"] = make_json_safe(text_res)

        # ---- Image inference ----
        if request.mode in ["image", "both"]:
            if not image_path.exists():
                entry_result["image"] = {
                    "error": f"Image not found: {image_path}"
                }
            else:
                image_res = run_image_inference(
                    image_input=str(image_path),
                    top_k=request.top_k
                )
                entry_result["image"] = make_json_safe(image_res)

        # ---- Combine ----
        if "text" in entry_result and "image" in entry_result:
            entry_result["combined"] = combine_probabilities(
                entry_result["image"].get("probabilities", {}),
                entry_result["text"].get("probabilities", {}),
                request.top_k
            )

        if "combined" in entry_result:
            entry_result["combined"] = make_json_safe(entry_result["combined"])

        results.append(entry_result)

    return results
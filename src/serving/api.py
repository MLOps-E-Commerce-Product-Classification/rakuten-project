from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import math
import numpy as np


from src.inference.run_image_inference import run_image_inference
from src.inference.run_text_inference import run_text_inference  


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
        return str(obj)  # fallback



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

# Pfade
CSV_PATH = "data/raw/X_test_update.csv"
IMAGE_DIR = "data/raw/images/image_test"
df = pd.read_csv(CSV_PATH, index_col=0)


class InferenceRequest(BaseModel):
    ids: List[int]           # IDs between 84916 and 98727
    mode: str = "both"       # "text", "image", "both"
    top_k: int = 27

@api.get('/')
def get_api():
    """Check whether API is running"""
    return {'Rakuten Inference API is up and running'}

@api.post("/inference")
def get_label(request: InferenceRequest):
    # load CSV
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail=f"{CSV_PATH} not found")

    results = []

    for id_ in request.ids:
        row = df.loc[id_]
        print(row)
        if id_ not in df.index:
            results.append({"id": id_, "error": "ID not found in CSV"})
            continue

        text_input = {"designation": row.get("designation", ""),
                "description": row.get("description", "")
        }
            
        productid = row["productid"]
        imageid = row["imageid"]
        image_filename = f"image_{imageid}_product_{productid}.jpg"
        image_path = os.path.join(IMAGE_DIR, image_filename)

        entry_result = {"id": id_}

        if request.mode in ["text", "both"]:
            text_res = run_text_inference(text_input=text_input, top_k=request.top_k)
            entry_result["text"] = make_json_safe(text_res)

        if request.mode in ["image", "both"]:
            if not os.path.exists(image_path):
                entry_result["image"] = {"error": f"Image not found: {image_path}"}
            else:
                image_res = run_image_inference(image_input=image_path, top_k=request.top_k)
                entry_result["image"] = make_json_safe(image_res)
                

        # Optional: combine text + image prediction
        if "text" in entry_result and "image" in entry_result:
            entry_result["combined"] = combine_probabilities(
                entry_result["image"]["probabilities"],
                entry_result["text"]["probabilities"],
                request.top_k
            )

        if "combined" in entry_result:
            entry_result["combined"] = make_json_safe(entry_result["combined"])
        
        results.append(entry_result)

    return results



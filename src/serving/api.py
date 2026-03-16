from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os

from src.inference.run_image_inference import run_image_inference
from src.inference.run_text_inference import run_text_inference  


def combine_probabilities(image_probs: dict, text_probs: dict):
    """
    Average probabilities per Rakuten code.
    """
    combined = {}
    all_codes = set(image_probs.keys()).union(text_probs.keys())
    for code in all_codes:
        combined[code] = (image_probs.get(code, 0) + text_probs.get(code, 0)) / 2
    # return top-K sorted
    top_k = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:5]
    return [{"rakuten_code": c, "probability": p} for c, p in top_k]


app = FastAPI(
    title="Rakuten Inference API",
    description="API to generate Rakuten label based on image and/or text using test CSV and images"
)

# Pfade
CSV_PATH = "data/raw/X_test_update.csv"
IMAGE_DIR = "data/raw/images/image_test"

class InferenceRequest(BaseModel):
    ids: List[int]           # IDs between 84916 and 98727
    mode: str = "both"       # "text", "image", "both"
    top_k: int = 27

@app.post("/inference")
def get_label(request: InferenceRequest):
    # load CSV
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail=f"{CSV_PATH} not found")
    df = pd.read_csv(CSV_PATH)

    results = []

    for id_ in request.ids:
        row = df[df["id"] == id_]
        if row.empty:
            results.append({"id": id_, "error": "ID not found in CSV"})
            continue

        row = row.iloc[0]
        text_input = row.get("designation", "") + " " + str(row.get("description", ""))

        productid = row["productid"]
        imageid = row["imageid"]
        image_filename = f"image_{imageid}_product_{productid}.jpg"
        image_path = os.path.join(IMAGE_DIR, image_filename)

        entry_result = {"id": id_}

        if request.mode in ["text", "both"]:
            text_res = run_text_inference(text_input=text_input, top_k=request.top_k)
            entry_result["text"] = text_res

        if request.mode in ["image", "both"]:
            if not os.path.exists(image_path):
                entry_result["image"] = {"error": f"Image not found: {image_path}"}
            else:
                image_res = run_image_inference(image_input=image_path, top_k=request.top_k)
                entry_result["image"] = image_res

        # Optional: combine text + image prediction
        if "text" in entry_result and "image" in entry_result:
            entry_result["combined"] = combine_probabilities(entry_result["image"], entry_result["text"])

        results.append(entry_result)

    return results



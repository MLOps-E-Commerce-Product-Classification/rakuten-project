from pathlib import Path
import json
import torch
import yaml

from src.models.image_classifier import build_image_model
from src.data.image_preprocessing import preprocess_image

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_encoding(label_encoding_path: str | Path) -> dict:
    label_encoding_path = Path(label_encoding_path)
    if not label_encoding_path.exists():
        raise FileNotFoundError(f"Label encoding file not found: {label_encoding_path}")
    with label_encoding_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_image_tensor(image_path: str | Path, preprocessing_config_path: str | Path) -> torch.Tensor:
    processed = preprocess_image(image=image_path, config_path=preprocessing_config_path)
    if isinstance(processed, tuple):
        image_array, _ = processed
    else:
        image_array = processed
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    return image_tensor


@torch.no_grad()
def predict_single_image_all(
    model: torch.nn.Module,
    image_path: str | Path,
    preprocessing_config_path: str | Path,
    device: torch.device,
    idx_to_code: dict[str, int],
    top_k: int = 5
) -> dict:

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_tensor = prepare_image_tensor(image_path, preprocessing_config_path).unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu()

    probabilities = torch.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    probabilities = probabilities / probabilities.sum()

    # Map probabilities to Rakuten codes
    code_prob = {
        int(idx_to_code[str(i)]): float(probabilities[i])
        for i in range(len(probabilities))
    }

    # Compute top_k
    sorted_codes = sorted(code_prob.items(), key=lambda x: x[1], reverse=True)
    top_k_predictions = [
        {"rakuten_code": int(code), "probability": float(prob)}
        for code, prob in sorted_codes[:top_k]
    ]

    predicted_rakuten_code = top_k_predictions[0]["rakuten_code"]

    return {
        "image_path": str(image_path),
        "predicted_rakuten_code": predicted_rakuten_code,
        "top_k_predictions": top_k_predictions,
        "probabilities": code_prob
    }


def predict_multiple_images_all(
    model: torch.nn.Module,
    image_paths: list[str | Path],
    preprocessing_config_path: str | Path,
    device: torch.device,
    idx_to_code: dict[str, int],
    top_k: int = 5
) -> list[dict]:

    return [
        predict_single_image_all(
            model=model,
            image_path=path,
            preprocessing_config_path=preprocessing_config_path,
            device=device,
            idx_to_code=idx_to_code,
            top_k=top_k
        )
        for path in image_paths
    ]


def save_inference_results(results: dict | list[dict], output_path: str | Path = RESULTS_PATH / "inference_results.json") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_image_inference(
    image_input: str | Path | list[str | Path],
    train_config_path: str | Path = "configs/image_train_config.yaml",
    preprocessing_config_path: str | Path = "configs/image_preprocessing_config.yaml",
    model_weights_path: str | Path = "models/best_image_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
    output_path: str | Path = "results/inference_results.json",
    top_k: int = 5,
) -> dict | list[dict]:
    # Load configs and label encoding
    train_config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)
    model_name = train_config.get("model", {}).get("name", "efficientnet_b0")
    idx_to_code = label_encoding["idx_to_code"]

    # Build model and load weights
    num_classes = len(label_encoding["classes"])
    model = build_image_model(model_name=model_name, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_weights_path = Path(model_weights_path)
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference
    if isinstance(image_input, (str, Path)):
        results = predict_single_image_all(
            model,
            image_input,
            preprocessing_config_path,
            device,
            idx_to_code,
            top_k
        )
    else:
        results = predict_multiple_images_all(
            model,
            image_input,
            preprocessing_config_path,
            device,
            idx_to_code,
            top_k
        )
        

    save_inference_results(results, output_path)
    return results


if __name__ == "__main__":
    results = run_image_inference(
        image_input=[
            "data/images/image_train/image_1263597046_product_3804725264.jpg",
            "data/images/image_train/image_1008141237_product_436067568.jpg",
        ],
        train_config_path="configs/image_train_config.yaml",
        preprocessing_config_path="configs/image_preprocessing_config.yaml",
        model_weights_path="models/best_image_model.pt",
        label_encoding_path="configs/label_encoding.json",
        output_path="results/inference_results.json",
        top_k=5
    )

    print("Inference finished. Results saved to: results/inference_results.json")
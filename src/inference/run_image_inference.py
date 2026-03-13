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
        raise FileNotFoundError(
            f"Label encoding file not found: {label_encoding_path}"
        )

    with label_encoding_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_image_tensor(
    image_path: str | Path,
    preprocessing_config_path: str | Path,
) -> torch.Tensor:
    processed = preprocess_image(
        image=image_path,
        config_path=preprocessing_config_path,
    )

    if isinstance(processed, tuple):
        image_array, _ = processed
    else:
        image_array = processed

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    return image_tensor


@torch.no_grad()
def predict_single_image(
    model: torch.nn.Module,
    image_path: str | Path,
    preprocessing_config_path: str | Path,
    device: torch.device,
    idx_to_code: dict[str, int],
    top_k: int = 5,
) -> dict:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_tensor = prepare_image_tensor(
        image_path=image_path,
        preprocessing_config_path=preprocessing_config_path,
    )

    image_tensor = image_tensor.unsqueeze(0).to(device)

    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)

    top_k = min(top_k, probabilities.shape[1])

    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_indices = top_indices.squeeze(0).cpu().tolist()

    top_predictions = []
    for class_idx, prob in zip(top_indices, top_probs):
        rakuten_code = idx_to_code[str(class_idx)]
        top_predictions.append(
            {
                "encoded_label": int(class_idx),
                "rakuten_code": int(rakuten_code),
                "probability": float(prob),
            }
        )

    predicted_class_idx = int(top_indices[0])
    predicted_rakuten_code = int(idx_to_code[str(predicted_class_idx)])

    return {
        "image_path": str(image_path),
        "predicted_encoded_label": predicted_class_idx,
        "predicted_rakuten_code": predicted_rakuten_code,
        "top_k_predictions": top_predictions,
    }


@torch.no_grad()
def predict_multiple_images(
    model: torch.nn.Module,
    image_paths: list[str | Path],
    preprocessing_config_path: str | Path,
    device: torch.device,
    idx_to_code: dict[str, int],
    top_k: int = 5,
) -> list[dict]:
    results = []

    for image_path in image_paths:
        result = predict_single_image(
            model=model,
            image_path=image_path,
            preprocessing_config_path=preprocessing_config_path,
            device=device,
            idx_to_code=idx_to_code,
            top_k=top_k,
        )
        results.append(result)

    return results


def save_inference_results(
    results: dict | list[dict],
    output_path: str | Path = RESULTS_PATH / "inference_results.json",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_image_inference(
    image_input: str | Path | list[str | Path],
    train_config_path: str | Path = "configs/train_config.yaml",
    preprocessing_config_path: str | Path = "configs/image_preprocessing_config.yaml",
    model_weights_path: str | Path = "models/best_image_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
    output_path: str | Path = "results/inference_results.json",
    top_k: int = 5,
) -> dict | list[dict]:
    train_config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)

    model_config = train_config.get("model", {})

    model_name = model_config.get("name", "efficientnet_b0")
    pretrained = False
    freeze_backbone = False

    num_classes = len(label_encoding["classes"])
    idx_to_code = label_encoding["idx_to_code"]

    model = build_image_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_weights_path = Path(model_weights_path)
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if isinstance(image_input, (str, Path)):
        results = predict_single_image(
            model=model,
            image_path=image_input,
            preprocessing_config_path=preprocessing_config_path,
            device=device,
            idx_to_code=idx_to_code,
            top_k=top_k,
        )
    else:
        results = predict_multiple_images(
            model=model,
            image_paths=image_input,
            preprocessing_config_path=preprocessing_config_path,
            device=device,
            idx_to_code=idx_to_code,
            top_k=top_k,
        )

    save_inference_results(results, output_path)

    return results


if __name__ == "__main__":
    results = run_image_inference(
        image_input=[
            "data/images/example_1.jpg",
            "data/images/example_2.jpg",
        ],
        train_config_path="configs/train_config.yaml",
        preprocessing_config_path="configs/image_preprocessing_config.yaml",
        model_weights_path="models/best_image_model.pt",
        label_encoding_path="configs/label_encoding.json",
        output_path="results/inference_results.json",
        top_k=5,
    )

    print("Inference finished.")
    print("Results saved to: results/inference_results.json")
from pathlib import Path
import json

import torch
import yaml

from image_model import build_image_model
from image_preprocessing import preprocess_image


RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_mapping(mapping_path: str | Path) -> dict[str, int]:
    mapping_path = Path(mapping_path)

    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def invert_label_mapping(label_mapping: dict[str, int]) -> dict[int, str]:
    return {int(encoded): original for original, encoded in label_mapping.items()}


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
    idx_to_label: dict[int, str],
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
        top_predictions.append(
            {
                "encoded_label": int(class_idx),
                "rakuten_code": idx_to_label[int(class_idx)],
                "probability": float(prob),
            }
        )

    predicted_class_idx = int(top_indices[0])
    predicted_label = idx_to_label[predicted_class_idx]

    return {
        "image_path": str(image_path),
        "predicted_encoded_label": predicted_class_idx,
        "predicted_rakuten_code": predicted_label,
        "top_k_predictions": top_predictions,
    }


@torch.no_grad()
def predict_multiple_images(
    model: torch.nn.Module,
    image_paths: list[str | Path],
    preprocessing_config_path: str | Path,
    device: torch.device,
    idx_to_label: dict[int, str],
    top_k: int = 5,
) -> list[dict]:
    results = []

    for image_path in image_paths:
        result = predict_single_image(
            model=model,
            image_path=image_path,
            preprocessing_config_path=preprocessing_config_path,
            device=device,
            idx_to_label=idx_to_label,
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
    label_mapping_path: str | Path = "artifacts/label_mapping.json",
    output_path: str | Path = "results/inference_results.json",
    top_k: int = 5,
) -> dict | list[dict]:
    train_config = load_config(train_config_path)
    label_mapping = load_label_mapping(label_mapping_path)
    idx_to_label = invert_label_mapping(label_mapping)

    model_config = train_config.get("model", {})

    model_name = model_config.get("name", "efficientnet_b0")
    pretrained = False
    freeze_backbone = False

    num_classes = len(label_mapping)

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
            idx_to_label=idx_to_label,
            top_k=top_k,
        )
    else:
        results = predict_multiple_images(
            model=model,
            image_paths=image_input,
            preprocessing_config_path=preprocessing_config_path,
            device=device,
            idx_to_label=idx_to_label,
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
        label_mapping_path="artifacts/label_mapping.json",
        output_path="results/inference_results.json",
        top_k=5,
    )

    print("Inference finished.")
    print("Results saved to: results/inference_results.json")
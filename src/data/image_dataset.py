from pathlib import Path

import torch
from torch.utils.data import Dataset

from image_preprocessing import preprocess_image
from src.data.label_encoding import load_label_encoding


class RakutenImageDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_dir: str | Path,
        config_path: str | Path,
        image_id_col: str = "image_id",
        label_col: str = "label",
        return_quality_report: bool = False,
        label_encoding_path = None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.config_path = config_path

        self.code_to_idx = None

        if label_encoding_path is not None:
            encoding = load_label_encoding(label_encoding_path)
            self.code_to_idx = encoding["code_to_idx"]

        self.image_id_col = image_id_col
        self.label_col = label_col
        self.return_quality_report = return_quality_report

        required_cols = {self.image_id_col, self.label_col}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = row["image_id"]
        label = row["rakuten_code"]

        if self.code_to_idx is not None:
            label = self.code_to_idx[str(label)]

        image_path = self.image_dir / f"{image_id}.jpg"

        processed = preprocess_image(
            image_path,
            image_id=image_id,
            config_path=self.config_path
        )

        quality_report = None
        if isinstance(processed, tuple):
            image, quality_report = processed
        else:
            image = processed

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        sample = {
            "image": image,
            "label": label,
            "image_id": image_id,
        }

        if self.return_quality_report and quality_report is not None:
            sample["quality_report"] = quality_report

        return sample
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.image_preprocessing import preprocess_image
from src.data.label_encoding import load_label_encoding


class RakutenImageDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_dir: str | Path,
        config_path: str | Path,
        image_id_col: str = "imageid",
        product_id_col: str = "productid",
        label_col: str = "prdtypecode",
        return_quality_report: bool = False,
        label_encoding_path: str | Path | None = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.config_path = config_path

        self.image_id_col = image_id_col
        self.product_id_col = product_id_col
        self.label_col = label_col
        self.return_quality_report = return_quality_report

        required_cols = {
            self.image_id_col,
            self.product_id_col,
            self.label_col,
        }
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.code_to_idx = None
        if label_encoding_path is not None:
            encoding = load_label_encoding(label_encoding_path)
            self.code_to_idx = encoding["code_to_idx"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image_id = str(row[self.image_id_col])
        product_id = str(row[self.product_id_col])
        label = row[self.label_col]

        if self.code_to_idx is not None:
            label = self.code_to_idx[str(label)]

        image_filename = f"image_{image_id}_product_{product_id}.jpg"
        image_path = self.image_dir / image_filename

        processed = preprocess_image(
            image_path,
            image_id=image_filename,
            config_path=self.config_path,
        )

        quality_report = None
        if isinstance(processed, tuple):
            image, quality_report = processed
        else:
            image = processed

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        sample = {
            "image": image,
            "label": int(label),
            "image_id": image_id,
            "product_id": product_id,
            "image_filename": image_filename,
        }

        if self.return_quality_report and quality_report is not None:
            sample["quality_report"] = quality_report

        return sample
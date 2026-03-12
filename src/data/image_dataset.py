from pathlib import Path

import torch
from torch.utils.data import Dataset

from image_preprocessing import preprocess_image


class RakutenImageDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_dir: str | Path,
        config_path: str | Path,
        image_id_col: str = "image_id",
        label_col: str = "label",
        return_quality_report: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.config_path = config_path
        self.image_id_col = image_id_col
        self.label_col = label_col
        self.return_quality_report = return_quality_report

        required_cols = {self.image_id_col, self.label_col}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image_id = str(row[self.image_id_col])
        label = int(row[self.label_col])

        image_path = self.image_dir / f"{image_id}.jpg"

        processed = preprocess_image(
            image_path,
            image_id=image_id,
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
            "label": label,
            "image_id": image_id,
        }

        if self.return_quality_report and quality_report is not None:
            sample["quality_report"] = quality_report

        return sample
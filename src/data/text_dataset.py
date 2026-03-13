from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.text_preprocessing import (
    build_tokenizer,
    load_text_preprocessing_config,
    preprocess_text,
)


class RakutenTextDataset(Dataset):
    def __init__(
        self,
        dataframe,
        config_path: str | Path,
        label_encoding: dict,  # Add this parameter
        designation_col: str = "designation",
        description_col: str = "description",
        label_col: str = "label",
        return_quality_report: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.label_map = label_encoding["code_to_idx"]
        self.config_path = Path(config_path)
        self.designation_col = designation_col
        self.description_col = description_col
        self.label_col = label_col
        self.return_quality_report = return_quality_report

        preprocessing_config = load_text_preprocessing_config(self.config_path)
        preprocessing_settings = preprocessing_config.get("preprocessing", {})

        self.max_length = int(preprocessing_settings.get("max_length", 128))
        self.tokenizer = build_tokenizer(self.config_path)

        required_cols = {self.designation_col, self.label_col}
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if self.description_col not in self.df.columns:
            self.df[self.description_col] = ""

    def __len__(self) -> int:
        return len(self.df)

    def _get_optional_metadata(self, row: Any) -> dict:
        metadata = {}

        for col in ["productid", "imageid", "Unnamed: 0"]:
            if col in row.index:
                value = row[col]
                if col == "Unnamed: 0":
                    metadata["sample_id"] = str(value)
                else:
                    metadata[col] = str(value)

        return metadata

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        designation = row[self.designation_col]
        description = row[self.description_col]
        raw_label = str(row[self.label_col])
        label = self.label_map[raw_label]

        processed = preprocess_text(
            designation=designation,
            description=description,
            config_path=self.config_path,
        )

        quality_report = None
        if isinstance(processed, tuple):
            text, quality_report = processed
        else:
            text = processed

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        sample = {  
            "input_ids": encoding["input_ids"].squeeze(0),  
            "attention_mask": encoding["attention_mask"].squeeze(0),  
            "label": torch.tensor(label, dtype=torch.long), # Now it's 0-N  
            "text": text,  
        }

        sample.update(self._get_optional_metadata(row))

        if self.return_quality_report and quality_report is not None:
            sample["quality_report"] = quality_report

        return sample

## unit tests for functions from text_dataset.py

from pathlib import Path
import pandas as pd
import pytest
import torch
from src.data import text_dataset as td


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "designation": "Chemise <b>bleue</b>",
                "description": "Coton premium",
                "prdtypecode": "100",
                "productid": 123,
                "imageid": 456,
                "Unnamed: 0": 789,
            }
        ]
    )


@pytest.fixture
def dataset(
    monkeypatch: pytest.MonkeyPatch,
    sample_dataframe: pd.DataFrame,
    text_preprocessing_config_path: Path,
    label_encoding: dict,
    dummy_tokenizer,
) -> td.RakutenTextDataset:
    monkeypatch.setattr(td, "build_tokenizer", lambda config_path: dummy_tokenizer)

    return td.RakutenTextDataset(
        dataframe=sample_dataframe,
        config_path=text_preprocessing_config_path,
        label_encoding=label_encoding,
        designation_col="designation",
        description_col="description",
        label_col="prdtypecode",
    )


def test_dataset_length_matches_number_of_rows(dataset: td.RakutenTextDataset):
    """The dataset length should mirror the source dataframe length."""
    assert len(dataset) == 1


def test_dataset_getitem_returns_tokenized_sample(dataset: td.RakutenTextDataset):
    """A dataset item should contain token tensors, encoded label, text and metadata."""
    sample = dataset[0]

    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["attention_mask"], torch.Tensor)
    assert isinstance(sample["label"], torch.Tensor)
    assert sample["label"].item() == 0
    assert sample["text"] == "Chemise bleue [SEP] Coton premium"
    assert sample["productid"] == "123"
    assert sample["imageid"] == "456"
    assert sample["sample_id"] == "789"
    assert sample["input_ids"].shape[0] == 8
    assert sample["attention_mask"].sum().item() > 0


def test_dataset_fills_missing_description_column(
    monkeypatch: pytest.MonkeyPatch,
    text_preprocessing_config_path: Path,
    label_encoding: dict,
    dummy_tokenizer,
):
    """If the description column is absent, the dataset should add an empty one."""
    dataframe = pd.DataFrame([{"designation": "Livre", "prdtypecode": "100"}])
    monkeypatch.setattr(td, "build_tokenizer", lambda config_path: dummy_tokenizer)

    dataset = td.RakutenTextDataset(
        dataframe=dataframe,
        config_path=text_preprocessing_config_path,
        label_encoding=label_encoding,
        designation_col="designation",
        description_col="description",
        label_col="prdtypecode",
    )

    assert "description" in dataset.df.columns
    assert dataset[0]["text"] == "Livre"


def test_dataset_returns_quality_report_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    sample_dataframe: pd.DataFrame,
    quality_text_preprocessing_config_path: Path,
    label_encoding: dict,
    dummy_tokenizer,
):
    """Quality reports should be attached when the dataset is asked to keep them."""
    monkeypatch.setattr(td, "build_tokenizer", lambda config_path: dummy_tokenizer)
    monkeypatch.setattr(
        td,
        "preprocess_text",
        lambda designation, description, config_path: (
            "texte nettoyé",
            {"very_short_text": False, "combined_word_count": 2},
        ),
    )

    dataset = td.RakutenTextDataset(
        dataframe=sample_dataframe,
        config_path=quality_text_preprocessing_config_path,
        label_encoding=label_encoding,
        designation_col="designation",
        description_col="description",
        label_col="prdtypecode",
        return_quality_report=True,
    )

    sample = dataset[0]

    assert sample["text"] == "texte nettoyé"
    assert sample["quality_report"] == {
        "very_short_text": False,
        "combined_word_count": 2,
    }


@pytest.mark.parametrize("missing_column", ["designation", "prdtypecode"])
def test_dataset_raises_when_required_columns_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    sample_dataframe: pd.DataFrame,
    text_preprocessing_config_path: Path,
    label_encoding: dict,
    dummy_tokenizer,
    missing_column: str,
):
    """The dataset should fail fast when a required column is absent."""
    monkeypatch.setattr(td, "build_tokenizer", lambda config_path: dummy_tokenizer)
    dataframe = sample_dataframe.drop(columns=[missing_column])

    with pytest.raises(ValueError, match="Missing required columns"):
        td.RakutenTextDataset(
            dataframe=dataframe,
            config_path=text_preprocessing_config_path,
            label_encoding=label_encoding,
            designation_col="designation",
            description_col="description",
            label_col="prdtypecode",
        )

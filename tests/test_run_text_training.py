## unit tests for functions from run_text_training.py

from pathlib import Path
import pandas as pd
import pytest
from src.training import run_text_training as rtt


@pytest.fixture
def training_dataframe() -> pd.DataFrame:
    rows = []
    for label in ["100", "200", "300"]:
        for idx in range(10):
            rows.append(
                {
                    "designation": f"Produit {label}-{idx}",
                    "description": f"Description {label}-{idx}",
                    "prdtypecode": label,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def rare_label_dataframe() -> pd.DataFrame:
    rows = []
    for label, count in [("100", 5), ("200", 5), ("300", 1)]:
        for idx in range(count):
            rows.append(
                {
                    "designation": f"Produit {label}-{idx}",
                    "description": f"Description {label}-{idx}",
                    "prdtypecode": label,
                }
            )
    return pd.DataFrame(rows)


def test_load_config_raises_on_missing_file(tmp_path: Path):
    """A missing training config should raise a clear FileNotFoundError."""
    missing_path = tmp_path / "missing_train_config.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        rtt.load_config(missing_path)


def test_load_label_encoding_raises_on_missing_file(tmp_path: Path):
    """A missing label encoding file should raise a clear FileNotFoundError."""
    missing_path = tmp_path / "missing_label_encoding.json"

    with pytest.raises(FileNotFoundError, match="Label encoding file not found"):
        rtt.load_label_encoding(missing_path)


def test_validate_dataframe_columns_raises_for_missing_columns():
    """Missing mandatory dataframe columns should raise a descriptive ValueError."""
    df = pd.DataFrame({"designation": ["robe"]})

    with pytest.raises(ValueError, match="missing required columns"):
        rtt.validate_dataframe_columns(df, {"designation", "prdtypecode"}, "Train DF")


def test_validate_labels_in_mapping_raises_for_unknown_labels():
    """Unknown labels should be rejected before training starts."""
    df = pd.DataFrame({"prdtypecode": ["100", "999"]})
    code_to_idx = {"100": 0}

    with pytest.raises(
        ValueError,
        match="not present in the predefined label encoding",
    ):
        rtt.validate_labels_in_mapping(df, "prdtypecode", code_to_idx, "Train DF")


def test_save_and_load_split_ids_roundtrip(tmp_path: Path):
    """Persisted split ids should round-trip without losing row order."""
    output_path = tmp_path / "splits" / "train_ids.csv"
    expected = pd.Index([7, 2, 9])

    rtt.save_split_ids(expected, output_path)
    loaded = rtt.load_split_ids(output_path)

    assert list(loaded) == [7, 2, 9]


def test_load_split_ids_raises_on_missing_file(tmp_path: Path):
    """Missing split files should raise a clear FileNotFoundError."""
    missing_path = tmp_path / "splits" / "train_ids.csv"

    with pytest.raises(FileNotFoundError, match="Split file not found"):
        rtt.load_split_ids(missing_path)


def test_load_split_ids_raises_when_row_id_column_is_missing(tmp_path: Path):
    """Malformed split files must contain a 'row_id' column."""
    split_path = tmp_path / "splits" / "train_ids.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"wrong_column": [1, 2, 3]}).to_csv(split_path, index=False)

    with pytest.raises(ValueError, match="must contain a 'row_id' column"):
        rtt.load_split_ids(split_path)


def test_load_or_create_splits_creates_stratified_non_overlapping_splits_and_persists_ids(
    training_dataframe: pd.DataFrame,
    tmp_path: Path,
):
    """A new split should create balanced, disjoint train/val/test partitions and persist their ids."""
    train_df, val_df, test_df = rtt.load_or_create_splits(
        df=training_dataframe,
        label_col="prdtypecode",
        split_ids_dir=tmp_path / "splits",
        seed=42,
        force_new_split=True,
        val_size=0.2,
        test_size=0.2,
    )

    assert len(train_df) == 18
    assert len(val_df) == 6
    assert len(test_df) == 6

    assert set(train_df.index).isdisjoint(val_df.index)
    assert set(train_df.index).isdisjoint(test_df.index)
    assert set(val_df.index).isdisjoint(test_df.index)

    assert train_df["prdtypecode"].value_counts().to_dict() == {
        "100": 6,
        "200": 6,
        "300": 6,
    }
    assert val_df["prdtypecode"].value_counts().to_dict() == {
        "100": 2,
        "200": 2,
        "300": 2,
    }
    assert test_df["prdtypecode"].value_counts().to_dict() == {
        "100": 2,
        "200": 2,
        "300": 2,
    }

    assert (tmp_path / "splits" / "train_ids.csv").exists()
    assert (tmp_path / "splits" / "val_ids.csv").exists()
    assert (tmp_path / "splits" / "test_ids.csv").exists()


def test_load_or_create_splits_reuses_saved_ids_when_available(
    training_dataframe: pd.DataFrame,
    tmp_path: Path,
):
    """Existing split files should be reused when force_new_split is false."""
    split_dir = tmp_path / "splits"
    first = rtt.load_or_create_splits(
        df=training_dataframe,
        label_col="prdtypecode",
        split_ids_dir=split_dir,
        seed=42,
        force_new_split=True,
        val_size=0.2,
        test_size=0.2,
    )

    second = rtt.load_or_create_splits(
        df=training_dataframe,
        label_col="prdtypecode",
        split_ids_dir=split_dir,
        seed=999,
        force_new_split=False,
        val_size=0.2,
        test_size=0.2,
    )

    for first_df, second_df in zip(first, second):
        assert list(first_df.index) == list(second_df.index)


@pytest.mark.parametrize(
    ("val_size", "test_size", "message"),
    [
        (0, 0.2, "val_size must be between 0 and 1."),
        (0.2, 0, "test_size must be between 0 and 1."),
        (0.6, 0.4, "val_size + test_size must be < 1."),
    ],
)
def test_load_or_create_splits_validates_split_sizes(
    training_dataframe: pd.DataFrame,
    tmp_path: Path,
    val_size: float,
    test_size: float,
    message: str,
):
    """Invalid split ratios should fail with an exception."""
    with pytest.raises(Exception):
        rtt.load_or_create_splits(
            df=training_dataframe,
            label_col="prdtypecode",
            split_ids_dir=tmp_path / "splits",
            seed=42,
            force_new_split=True,
            val_size=val_size,
            test_size=test_size,
        )


def test_load_or_create_splits_falls_back_to_non_stratified_when_class_count_is_too_small(
    rare_label_dataframe: pd.DataFrame,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """If any class has fewer than two samples, the split should fall back to non-stratified mode."""
    train_df, val_df, test_df = rtt.load_or_create_splits(
        df=rare_label_dataframe,
        label_col="prdtypecode",
        split_ids_dir=tmp_path / "splits",
        seed=42,
        force_new_split=True,
        val_size=0.2,
        test_size=0.2,
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(rare_label_dataframe)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0


def test_load_or_create_splits_raises_when_saved_ids_are_not_in_dataframe(
    training_dataframe: pd.DataFrame,
    tmp_path: Path,
):
    """Reusing persisted splits should fail clearly if the saved row ids are absent from the dataframe."""
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"row_id": [999]}).to_csv(split_dir / "train_ids.csv", index=False)
    pd.DataFrame({"row_id": [0]}).to_csv(split_dir / "val_ids.csv", index=False)
    pd.DataFrame({"row_id": [1]}).to_csv(split_dir / "test_ids.csv", index=False)

    with pytest.raises(KeyError):
        rtt.load_or_create_splits(
            df=training_dataframe,
            label_col="prdtypecode",
            split_ids_dir=split_dir,
            seed=42,
            force_new_split=False,
            val_size=0.2,
            test_size=0.2,
        )


def test_run_text_training_orchestrates_pipeline_and_returns_expected_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The main training entry point should wire together config loading, datasets, loaders, model building and training."""
    model_save_path = tmp_path / "best_text_model.pt"

    config = {
        "training": {"batch_size": 4, "num_workers": 2, "seed": 123, "subset": 0},
        "model": {
            "name": "dummy-backbone",
            "pretrained": False,
            "freeze_backbone": True,
        },
        "data": {
            "label_col": "prdtypecode",
            "designation_col": "designation",
            "description_col": "description",
            "return_quality_report": True,
        },
    }
    label_encoding = {"code_to_idx": {"100": 0, "200": 1}, "classes": ["100", "200"]}

    captured: dict = {}

    (tmp_path / "train.csv").touch()
    (tmp_path / "val.csv").touch()

    # 1. Mock config
    monkeypatch.setattr(rtt, "load_config", lambda _: config)
    monkeypatch.setattr(rtt, "load_label_encoding", lambda _: label_encoding)
    monkeypatch.setattr(rtt, "set_seed", lambda seed: captured.setdefault("seed", seed))

    import torch
    import pandas as pd

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    if hasattr(rtt, "mlflow"):
        import unittest.mock

        mock_mlflow = unittest.mock.MagicMock()
        mock_mlflow.active_run.return_value.info.run_id = "mock-run-id"
        monkeypatch.setattr(rtt, "mlflow", mock_mlflow)

    # 2. Mock data loading
    fake_df = pd.DataFrame(
        {"designation": ["a"], "description": ["b"], "prdtypecode": ["100"]}
    )
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_df.copy())

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return idx

    monkeypatch.setattr(rtt, "RakutenTextDataset", DummyDataset)
    monkeypatch.setattr(rtt, "DataLoader", lambda *args, **kwargs: ["dummy_batch"])

    # 3. Mock training
    class DummyModel:
        def to(self, device):
            pass

    monkeypatch.setattr(rtt, "build_text_model", lambda *args, **kwargs: DummyModel())

    def fake_train_model(*args, **kwargs):
        return DummyModel(), {
            "val_macro_f1": [0.75],
            "val_accuracy": [0.80],
            "val_loss": [0.42],
        }

    monkeypatch.setattr(rtt, "train_model", fake_train_model)

    # 4. mock execution
    history, returned_label_encoding = rtt.run_text_training(
        processed_data_dir=tmp_path,
        train_config_path="ignored_train_config.yaml",
        preprocessing_config_path="prep_config.yaml",
        model_save_path=model_save_path,
        label_encoding_path="ignored_label_encoding.json",
    )

    assert returned_label_encoding == label_encoding
    assert history["val_macro_f1"] == [0.75]
    assert captured.get("seed") == 123

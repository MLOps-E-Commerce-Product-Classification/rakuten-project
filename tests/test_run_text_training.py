## unit tests for functions from run_text_training.py

from pathlib import Path
import re
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
    """Invalid split ratios should fail with a precise validation message."""
    with pytest.raises(ValueError, match=re.escape(message)):
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

    captured = capsys.readouterr()

    assert len(train_df) + len(val_df) + len(test_df) == len(rare_label_dataframe)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert "Falling back to non-stratified split" in captured.out


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


def test_run_text_training_raises_when_x_and_y_have_different_row_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Training should fail early if X and Y CSVs have different numbers of rows."""
    x_path = tmp_path / "X.csv"
    y_path = tmp_path / "Y.csv"

    pd.DataFrame(
        {
            "designation": ["a", "b"],
            "description": ["desc a", "desc b"],
        }
    ).to_csv(x_path, index=False)

    pd.DataFrame({"prdtypecode": ["100"]}).to_csv(y_path, index=False)

    monkeypatch.setattr(
        rtt,
        "load_config",
        lambda _: {"training": {}, "model": {}, "data": {}, "split": {}},
    )
    monkeypatch.setattr(
        rtt,
        "load_label_encoding",
        lambda _: {"code_to_idx": {"100": 0}, "classes": ["100"]},
    )

    with pytest.raises(ValueError, match="must have the same number of rows"):
        rtt.run_text_training(
            x_data_csv_path=x_path,
            y_data_csv_path=y_path,
            split_ids_dir=tmp_path / "splits",
        )


def test_run_text_training_orchestrates_pipeline_and_returns_expected_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The main training entry point should wire together config loading, splitting, datasets, loaders, model building and training."""
    x_path = tmp_path / "X.csv"
    y_path = tmp_path / "Y.csv"
    model_save_path = tmp_path / "best_text_model.pt"

    x_df = pd.DataFrame(
        {
            "designation": ["a", "b", "c", "d"],
            "description": ["desc a", "desc b", "desc c", "desc d"],
        }
    )
    y_df = pd.DataFrame({"prdtypecode": ["100", "200", "100", "200"]})

    x_df.to_csv(x_path, index=False)
    y_df.to_csv(y_path, index=False)

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
        "split": {"val_size": 0.2, "test_size": 0.2},
    }
    label_encoding = {
        "code_to_idx": {"100": 0, "200": 1},
        "classes": ["100", "200"],
    }

    captured: dict = {}

    monkeypatch.setattr(rtt, "load_config", lambda _: config)
    monkeypatch.setattr(rtt, "load_label_encoding", lambda _: label_encoding)
    monkeypatch.setattr(rtt, "set_seed", lambda seed: captured.setdefault("seed", seed))
    monkeypatch.setattr(rtt.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        rtt, "mlflow", __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    )

    def fake_load_or_create_splits(
        df: pd.DataFrame,
        label_col: str,
        split_ids_dir: str | Path,
        seed: int,
        force_new_split: bool,
        val_size: float,
        test_size: float,
    ):
        captured["split_call"] = {
            "label_col": label_col,
            "split_ids_dir": Path(split_ids_dir),
            "seed": seed,
            "force_new_split": force_new_split,
            "val_size": val_size,
            "test_size": test_size,
            "df_columns": list(df.columns),
        }
        return df.iloc[:2].copy(), df.iloc[2:3].copy(), df.iloc[3:].copy()

    monkeypatch.setattr(rtt, "load_or_create_splits", fake_load_or_create_splits)

    class DummyDataset:
        def __init__(
            self,
            dataframe,
            config_path,
            label_encoding,
            designation_col,
            description_col,
            label_col,
            return_quality_report,
        ):
            self.dataframe = dataframe.copy()
            captured.setdefault("datasets", []).append(
                {
                    "rows": len(dataframe),
                    "config_path": Path(config_path),
                    "designation_col": designation_col,
                    "description_col": description_col,
                    "label_col": label_col,
                    "return_quality_report": return_quality_report,
                    "label_encoding": label_encoding,
                }
            )

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            return idx

    monkeypatch.setattr(rtt, "RakutenTextDataset", DummyDataset)

    def fake_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
        captured.setdefault("dataloaders", []).append(
            {
                "rows": len(dataset),
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
            }
        )
        return {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    monkeypatch.setattr(rtt, "DataLoader", fake_dataloader)

    def fake_build_text_model(model_name, num_classes, pretrained, freeze_backbone):
        captured["build_model_call"] = {
            "model_name": model_name,
            "num_classes": num_classes,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
        }
        return "dummy-model"

    monkeypatch.setattr(rtt, "build_text_model", fake_build_text_model)

    def fake_train_model(
        model,
        train_dataloader,
        val_dataloader,
        config_path,
        num_classes,
        model_save_path,
    ):
        captured["train_model_call"] = {
            "model": model,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "config_path": config_path,
            "num_classes": num_classes,
            "model_save_path": Path(model_save_path),
        }
        return model, {
            "val_macro_f1": [0.75],
            "val_accuracy": [0.80],
            "val_loss": [0.42],
        }

    monkeypatch.setattr(rtt, "train_model", fake_train_model)

    trained_model, history, returned_label_encoding = rtt.run_text_training(
        x_data_csv_path=x_path,
        y_data_csv_path=y_path,
        split_ids_dir=tmp_path / "splits",
        force_new_split=True,
        train_config_path="ignored_train_config.yaml",
        preprocessing_config_path="prep_config.yaml",
        model_save_path=model_save_path,
        label_encoding_path="ignored_label_encoding.json",
    )

    assert trained_model == "dummy-model"
    assert returned_label_encoding == label_encoding
    assert history["val_macro_f1"] == [0.75]
    assert history["val_accuracy"] == [0.80]
    assert history["val_loss"] == [0.42]
    assert history["split_sizes"] == {"train": 2, "val": 1, "test": 1}

    assert captured["seed"] == 123
    assert captured["split_call"]["label_col"] == "prdtypecode"
    assert captured["split_call"]["seed"] == 123
    assert captured["split_call"]["force_new_split"] is True
    assert captured["split_call"]["val_size"] == 0.2
    assert captured["split_call"]["test_size"] == 0.2
    assert {"designation", "description", "prdtypecode"}.issubset(
        captured["split_call"]["df_columns"]
    )

    assert len(captured["datasets"]) == 2
    assert captured["datasets"][0]["rows"] == 2
    assert captured["datasets"][1]["rows"] == 1
    assert captured["datasets"][0]["config_path"] == Path("prep_config.yaml")
    assert captured["datasets"][0]["designation_col"] == "designation"
    assert captured["datasets"][0]["description_col"] == "description"
    assert captured["datasets"][0]["label_col"] == "prdtypecode"
    assert captured["datasets"][0]["return_quality_report"] is True

    assert captured["dataloaders"] == [
        {
            "rows": 2,
            "batch_size": 4,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": False,
        },
        {
            "rows": 1,
            "batch_size": 4,
            "shuffle": False,
            "num_workers": 2,
            "pin_memory": False,
        },
    ]

    assert captured["build_model_call"] == {
        "model_name": "dummy-backbone",
        "num_classes": 2,
        "pretrained": False,
        "freeze_backbone": True,
    }

    assert captured["train_model_call"]["model"] == "dummy-model"
    assert captured["train_model_call"]["config_path"] == "ignored_train_config.yaml"
    assert captured["train_model_call"]["num_classes"] == 2
    assert captured["train_model_call"]["model_save_path"] == model_save_path

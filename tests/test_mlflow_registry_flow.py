from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.serving import mlflow_bento as mb
from src.serving.promote_mlflow_model import promote_model
from src.serving.sync_mlflow_to_bento import sync_mlflow_model_to_bento
from src.training.mlflow_text_registry import register_text_model_in_mlflow


class FakeRunData:
    def __init__(self, metrics):
        self.metrics = metrics


class FakeRun:
    def __init__(self, metrics):
        self.data = FakeRunData(metrics)


class FakeMlflowClient:
    def __init__(self):
        self.versions = {}
        self.aliases = {}
        self.runs = {}
        self.model_version_tags = {}
        self.promotions = []

    def get_model_version_by_alias(self, name, alias):
        version = self.aliases[(name, alias)]
        return self.versions[(name, str(version))]

    def get_model_version(self, name, version):
        return self.versions[(name, str(version))]

    def search_model_versions(self, query):
        return list(self.versions.values())

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[(name, alias)] = str(version)
        self.promotions.append((name, alias, str(version)))

    def get_run(self, run_id):
        return self.runs[run_id]

    def set_model_version_tag(self, name, version, key, value):
        self.model_version_tags[(name, str(version), key)] = value


class FakePyFuncApi:
    def __init__(self):
        self.calls = []

    def log_model(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            registered_model_version="12",
            model_uri="runs:/run-123/text_classifier_model",
        )


class FakeMlflowModule:
    def __init__(self):
        self.pyfunc = FakePyFuncApi()
        self.logged_dicts = []
        self._active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    def active_run(self):
        return self._active_run

    def log_dict(self, payload, artifact_path):
        self.logged_dicts.append((payload, artifact_path))

    def set_tracking_uri(self, value):
        self.tracking_uri = value

    def set_registry_uri(self, value):
        self.registry_uri = value

    def MlflowClient(self):
        raise AssertionError("Client creation should be injected in this test")


class FakeBentoModel:
    def __init__(self, tag, metadata=None, labels=None):
        self.tag = tag
        self.metadata = metadata or {}
        self.labels = labels or {}


class FakeBentoMlflowApi:
    def __init__(self, store):
        self.store = store
        self.import_calls = []

    def import_model(self, name, model_uri, labels=None, metadata=None):
        tag = f"{name}:import-{len(self.import_calls) + 1}"
        model = FakeBentoModel(tag=tag, metadata=metadata, labels=labels)
        self.store.append(model)
        self.import_calls.append((name, model_uri, labels, metadata))
        return model


class FakeBentoModelsApi:
    def __init__(self, store):
        self.store = store

    def list(self, tag=None):
        if tag is None:
            return list(self.store)
        prefix = f"{tag}:"
        return [model for model in self.store if str(model.tag).startswith(prefix)]


class FakeBentoModule:
    def __init__(self):
        self.store = []
        self.models = FakeBentoModelsApi(self.store)
        self.mlflow = FakeBentoMlflowApi(self.store)


def test_register_text_model_in_mlflow_logs_pyfunc_and_manifest(tmp_path):
    fake_mlflow = FakeMlflowModule()
    fake_client = FakeMlflowClient()

    train_config = tmp_path / "train.yaml"
    preprocessing_config = tmp_path / "prep.yaml"
    label_encoding = tmp_path / "labels.json"
    weights = tmp_path / "model.pt"
    backbone = tmp_path / "backbone"
    backbone.mkdir()

    train_config.write_text("model:\n  name: dummy-backbone\n", encoding="utf-8")
    preprocessing_config.write_text("preprocessing:\n  tokenizer_model: dummy-backbone\n", encoding="utf-8")
    label_encoding.write_text('{"classes": ["100"], "code_to_idx": {"100": 0}}', encoding="utf-8")
    weights.write_bytes(b"fake-weights")

    manifest = register_text_model_in_mlflow(
        model_weights_path=weights,
        train_config_path=train_config,
        preprocessing_config_path=preprocessing_config,
        label_encoding_path=label_encoding,
        registered_model_name="rakuten_text_classifier",
        registration_manifest_path=tmp_path / "training_manifest.json",
        mlflow_module=fake_mlflow,
        client=fake_client,
        backbone_dir=backbone,
    )

    assert manifest["mlflow_model_name"] == "rakuten_text_classifier"
    assert manifest["mlflow_version"] == "12"
    assert manifest["mlflow_run_id"] == "run-123"
    assert (tmp_path / "training_manifest.json").exists()

    call = fake_mlflow.pyfunc.calls[0]
    assert call["registered_model_name"] == "rakuten_text_classifier"
    assert Path(call["artifacts"]["weights"]) == weights.resolve()
    assert Path(call["artifacts"]["backbone"]) == backbone.resolve()
    assert fake_client.model_version_tags[("rakuten_text_classifier", "12", "validation_status")] == "pending"


def test_promote_model_uses_metric_margin_and_alias_update():
    client = FakeMlflowClient()
    candidate = SimpleNamespace(version="12", run_id="run-12", tags={"validation_status": "approved"})
    champion = SimpleNamespace(version="11", run_id="run-11", tags={"validation_status": "approved"})
    client.versions[("rakuten_text_classifier", "12")] = candidate
    client.versions[("rakuten_text_classifier", "11")] = champion
    client.aliases[("rakuten_text_classifier", "champion")] = "11"
    client.runs["run-12"] = FakeRun({"eval_macro_f1": 0.83})
    client.runs["run-11"] = FakeRun({"eval_macro_f1": 0.80})

    result = promote_model(
        model_name="rakuten_text_classifier",
        alias="champion",
        candidate_version="12",
        metric_name="eval_macro_f1",
        min_improvement=0.01,
        required_tag="validation_status=approved",
        client=client,
    )

    assert result["promoted"] is True
    assert client.promotions == [("rakuten_text_classifier", "champion", "12")]
    assert result["candidate_metric_value"] == 0.83
    assert result["champion_metric_value"] == 0.80


def test_sync_mlflow_model_to_bento_is_idempotent_and_writes_manifest(tmp_path):
    client = FakeMlflowClient()
    client.versions[("rakuten_text_classifier", "12")] = SimpleNamespace(
        version="12",
        run_id="run-12",
        tags={"validation_status": "approved"},
    )
    client.aliases[("rakuten_text_classifier", "champion")] = "12"
    bentoml_module = FakeBentoModule()
    manifest_path = tmp_path / "deployment_manifest.json"

    first = sync_mlflow_model_to_bento(
        model_name="rakuten_text_classifier",
        alias="champion",
        manifest_path=str(manifest_path),
        client=client,
        bentoml_module=bentoml_module,
    )
    second = sync_mlflow_model_to_bento(
        model_name="rakuten_text_classifier",
        alias="champion",
        manifest_path=str(manifest_path),
        client=client,
        bentoml_module=bentoml_module,
    )

    assert first["updated"] is True
    assert second["updated"] is False
    assert first["bentoml_model_tag"] == second["bentoml_model_tag"]
    assert bentoml_module.mlflow.import_calls[0][1] == "models:/rakuten_text_classifier@champion"
    manifest_payload = mb.read_json_file(manifest_path)
    assert manifest_payload["mlflow_version"] == "12"
    assert manifest_payload["validation_status"] == "approved"


def test_resolve_bento_model_reference_prefers_env_then_manifest(tmp_path):
    manifest_path = tmp_path / "deployment_manifest.json"
    mb.write_json_file(
        manifest_path,
        {
            "bentoml_model_tag": "rakuten_text_classifier:import-7",
            "mlflow_model_name": "rakuten_text_classifier",
            "mlflow_alias": "champion",
            "mlflow_version": "12",
            "mlflow_run_id": "run-12",
            "validation_status": "approved",
        },
    )

    env = {
        "BENTO_DEPLOYMENT_MANIFEST": str(manifest_path),
    }
    resolved = mb.resolve_bento_model_reference(env=env)
    assert resolved["model_tag"] == "rakuten_text_classifier:import-7"
    assert resolved["mlflow_version"] == "12"

    env["BENTO_MODEL_TAG"] = "rakuten_text_classifier:manual"
    overridden = mb.resolve_bento_model_reference(env=env)
    assert overridden["model_tag"] == "rakuten_text_classifier:manual"

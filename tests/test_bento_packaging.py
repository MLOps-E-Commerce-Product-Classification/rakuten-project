from pathlib import Path

import yaml


def test_bentofile_targets_text_bento_service():
    bentofile = yaml.safe_load(Path("bentofile.yaml").read_text())
    assert bentofile["service"] == "src.serving.bento_service:TextBentoService"
    assert bentofile["models"] == ["rakuten_text_classifier:latest"]
    assert "configs/text_preprocessing_config.yaml" in bentofile["include"]
    assert "artifacts/deployment_manifest.json" in bentofile["include"]
    assert "mlflow>=2.22.1" in bentofile["python"]["packages"]


def test_compose_uses_bento_image_runtime():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())
    service = compose["services"]["bento-text-service"]
    assert service["image"] == "rakuten_text_service:latest"
    assert "build" not in service


def test_obsolete_source_serving_dockerfile_removed():
    assert not Path("docker/Dockerfile.api").exists()


def test_makefile_syncs_model_before_building_bento():
    makefile = Path("Makefile").read_text()
    assert "register-bento-text-model: sync-bento-text-model" in makefile
    assert "build-bento-text: sync-bento-text-model" in makefile
    assert "promote-mlflow-text-model:" in makefile
    assert "sync-bento-text-model:" in makefile


def test_backbone_vendoring_is_restricted_to_lightweight_assets():
    content = Path("src/serving/prepare_bento_assets.py").read_text()
    assert "allow_patterns=ALLOWED_BACKBONE_PATTERNS" in content
    assert '"*.bin"' in content
    assert '"*.safetensors"' in content

from pathlib import Path

import yaml
import pytest


def test_bentofile_targets_text_bento_service():
    bentofile = yaml.safe_load(Path("bentofile.yaml").read_text())
    assert bentofile["service"] == "src.serving.bento_service:TextBentoService"
    assert bentofile["models"] == ["text-classifier:latest"]
    assert "configs/text_preprocessing_config.yaml" in bentofile["include"]
    assert "artifacts/deployment_manifest.json" in bentofile["include"]
    assert any(pkg.startswith("mlflow") for pkg in bentofile["python"]["packages"])


def test_compose_uses_bento_image_runtime():
    compose_path = Path("docker-compose.yml")
    if not compose_path.exists():
        pytest.skip("docker-compose.yml ist aktuell nicht vorhanden")

    compose = yaml.safe_load(compose_path.read_text())

    service = compose["services"]["bento-text-service"]
    assert service["image"] == "rakuten-text-service:latest"
    assert "build" not in service


def test_obsolete_source_serving_dockerfile_removed():
    assert not Path("docker/Dockerfile.api").exists()


def test_makefile_syncs_model_before_building_bento():
    makefile = Path("Makefile").read_text()
    assert "build-bento: sync-bento" in makefile
    assert "promote-model:" in makefile
    assert "sync-bento:" in makefile


def test_backbone_vendoring_is_restricted_to_lightweight_assets():
    content = Path("src/serving/prepare_bento_assets.py").read_text()
    assert "allow_patterns=ALLOWED_BACKBONE_PATTERNS" in content
    assert '"*.bin"' in content
    assert '"*.safetensors"' in content

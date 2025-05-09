# test_enhance_images.py
# Pytest-compatible test for enhance_images_in_oid()

import pytest
from utils.enhance_images import enhance_images_in_oid
from utils.config_loader import resolve_config

# === Configurable Inputs ===
CONFIG_FILE = r"F:\RMI Mosaic 360 Tools Test AGP\Project\config.yaml"
PROJECT_FOLDER = r"F:\RMI Mosaic 360 Tools Test AGP\Project"
OID_FC = r"F:\RMI Mosaic 360 Tools Test AGP\Project\backups\oid_snapshots.gdb\reel0008_oid_test3_before_enhance_images_20250508_1751"

@pytest.mark.integration
def test_enhance_images_dry_run():
    messages = []

    config = resolve_config(
        config_file=CONFIG_FILE,
        project_folder=PROJECT_FOLDER,
        messages=messages
    )
    config["image_enhancement"]["dry_run"] = True

    result = enhance_images_in_oid(
        oid_fc_path=OID_FC,
        config=config,
        messages=messages
    )

    assert isinstance(result, dict)
    assert len(result) > 0

    print("\n✅ Dry run test completed with:")
    for k, v in result.items():
        print(f"{k}: {v}")

    if messages:
        print("\nMessages:")
        for m in messages:
            print("-", m)

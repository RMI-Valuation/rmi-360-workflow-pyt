# =============================================================================
# 🧪 Test Script: enhance_images_in_oid (test_enhance_images.py)
# -----------------------------------------------------------------------------
# Purpose:     Integration test for the dry-run mode of the image enhancement pipeline.
#
# Description:
#   - Loads config from a test project folder
#   - Runs enhance_images_in_oid() in dry-run mode
#   - Confirms the output is a dictionary and optionally prints mapped paths
#   - Validates that keys and values are plausible .jpg filenames
#
# Framework:   Pytest
# Author:      RMI Valuation, LLC
# Created:     2025-05-10
# =============================================================================

# test_enhance_images.py
# Pytest-compatible test for enhance_images_in_oid()

import pytest
from utils.enhance_images import enhance_images_in_oid
from utils.config_loader import resolve_config

# === Configurable Inputs ===
CONFIG_FILE = r"F:\RMI Mosaic 360 Tools Test AGP\Project\config.yaml"
PROJECT_FOLDER = r"F:\RMI Mosaic 360 Tools Test AGP\Project"
OID_FC = r"F:\RMI Mosaic 360 Tools Test AGP\Project\backups\oid_snapshots.gdb\reel0008_oid_test5_before_enhance_images_20250510_0738"


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

    assert isinstance(result, dict), "Output should be a dictionary"
    if not config["image_enhancement"].get("dry_run", False):
        assert len(result) > 0, "Expected non-empty output in non-dry-run mode"

    print("\n✅ Dry run test completed with:")

    if result:
        for k, v in result.items():
            assert isinstance(k, str) and k.endswith('.jpg'), f"Unexpected key: {k}"
            assert isinstance(v, str) and v.endswith('.jpg'), f"Unexpected value: {v}"
        for k, v in result.items():
            print(f"{k}: {v}")

    if messages:
        print("\nMessages:")
        for m in messages:
            print("-", m)

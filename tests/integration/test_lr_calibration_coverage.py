"""Every trainable Tinker model should have an LR calibration.

When this fails after a new model launches, run
``python -m tinker_cookbook.lr_calibration --missing``.
"""

import json

import tinker

from tinker_cookbook.lr_calibration.coverage import CALIBRATION_FILE, missing_models


def test_every_trainable_model_is_calibrated():
    table = json.loads(CALIBRATION_FILE.read_text())
    assert missing_models(table, tinker.RestClient()) == []

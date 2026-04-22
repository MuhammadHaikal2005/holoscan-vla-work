"""Modality config for the SO-101 arm + IMX274 front camera setup.

Differences from the stock SO100 config:
  - Only ONE camera: "front" (no wrist camera)
  - 6 joints split as: single_arm (0:5) + gripper (5:6)
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

so101_imx274_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front"],          # IMX274 front-view only, no wrist cam
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",                 # joints 0-4 (shoulder_pan … wrist_roll)
            "gripper",                    # joint 5
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),    # 16-step action chunk (same as SO100)
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    so101_imx274_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)

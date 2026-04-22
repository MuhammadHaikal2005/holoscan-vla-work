"""Modality config for the SO-101 arm + dual USB cameras setup.

Two cameras (cam0 and cam1) are mounted at different angles to give
the model depth/perspective cues that a single front-facing camera lacks.

Dataset feature keys:
  observation.images.cam0   — first USB camera  (/dev/video0)
  observation.images.cam1   — second USB camera (/dev/video2)
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

so101_dual_usb_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam0", "cam1"],   # both USB camera views
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",                 # joints 0-4 (shoulder_pan … wrist_roll)
            "gripper",                    # joint 5
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),    # 16-step action chunk
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
    so101_dual_usb_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)

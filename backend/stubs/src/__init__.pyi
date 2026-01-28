# Stub package to allow static type-checking of `src` modules in tests.
# Individual submodule stubs live alongside this file.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import python_feature_extractor  # type: ignore
    from . import runtime_core_bridge  # type: ignore
    from . import control_model  # type: ignore
    from . import visual_metrics  # type: ignore
    from . import data_loader  # type: ignore
    from . import control_trainer  # type: ignore
    from . import song_analyzer  # type: ignore

__all__ = [
    "python_feature_extractor",
    "runtime_core_bridge",
    "control_model",
    "visual_metrics",
    "data_loader",
    "control_trainer",
    "song_analyzer",
]

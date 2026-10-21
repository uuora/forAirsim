"""Load a bounded Bézier parameter profile without importing a search engine."""
from __future__ import annotations

from collections.abc import Mapping

from .core import BezierParameters


def load_trajectory_profile(config: Mapping) -> tuple[BezierParameters, str]:
    """Absent profiles preserve the analytical baseline; malformed ones fail early.

    Provenance in a file is descriptive, never a runtime safety certification.
    Search and neural-network dependencies belong to the offline optimizer.
    """
    section = config.get("trajectory", {})
    if not isinstance(section, Mapping):
        raise ValueError("trajectory must be an object")
    parameters = BezierParameters.from_dict(section.get("parameters", {}))
    name = section.get("name", "默认解析参数")
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise ValueError("trajectory.name must be a nonempty string of at most 100 characters")
    return parameters, name

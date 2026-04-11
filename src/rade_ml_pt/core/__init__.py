"""rade_ml_pt.core -- populated after porting."""
from typing import Any


def json_safe(obj: Any) -> Any:
    """Fallback serialiser for ``json.dump(..., default=json_safe)``.

    Converts non-serializable objects (class instances, numpy types, etc.)
    to a JSON-friendly representation so that ``json.dump`` never crashes
    on runtime-only objects like asset classes stored in metadata.
    """
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return f"<non-serializable: {type(obj).__name__}>"

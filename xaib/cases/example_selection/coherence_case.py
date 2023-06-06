from typing import Any

from ...base import Case, Dataset, Model
from ...metrics.example_selection import SameClassCheck


class CoherenceCase(Case):
    def __init__(self, ds: Dataset, model: Model, *args: Any, **kwargs: Any) -> None:
        super().__init__(ds, model, *args, **kwargs)
        self.name = "coherence"
        self._metric_objs["same_class_check"] = SameClassCheck(ds, model)

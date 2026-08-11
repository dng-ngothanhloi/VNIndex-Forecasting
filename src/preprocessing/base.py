"""
base.py – BasePreprocessor abstract contract (src/preprocessing)
===================================================================
Governed contract scoped exclusively to the fitted Z-score scaling step
implemented today by scale_by_train_stats in helpers/utils.py.

Does NOT cover raw loading, pre-split cleaning (steps 2-7 of
src/preprocess_steps/), or the time-based split — those remain plain
function calls and the Split_Boundary data contract, respectively,
invoked by the Pipeline_Shell (src/preprocess.py) outside this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class NotFittedError(RuntimeError):
    """Raised when a BasePreprocessor method that requires a prior fit(X)
    is called before fit(X) has been called successfully.

    A plain RuntimeError subclass defined in this package, mirroring
    src/reduction's NotFittedError precedent (per pca-reducer-wrapper's
    design), so BasePreprocessor's contract does not depend on any
    specific backing library's exception hierarchy.
    """


class BasePreprocessor(ABC):
    """Governed contract for the fitted Z-score scaling transformation.

    Scoped exclusively to the step implemented today by
    scale_by_train_stats in helpers/utils.py. Does NOT cover raw loading,
    pre-split cleaning (steps 2-7), or the time-based split — those remain
    plain function calls and the Split_Boundary data contract,
    respectively, invoked by the Pipeline_Shell outside this class.

    Subclasses MUST implement fit, transform, and get_metadata. Instantiating
    a subclass that leaves any of these abstract raises TypeError (enforced
    by abc.ABC + @abstractmethod), satisfying Requirement 1.6.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BasePreprocessor":
        """Fit on X (the Split_Boundary's train DataFrame only). Returns self."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform X using state stored during fit."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return the fitted preprocessor's metadata contract as a dict.

        Included for interface consistency with the Phase 1A BaseReducer
        precedent (a design choice, not itself mandated by the Master
        Roadmap's Req 7.1 — documented per Requirement 1.10).
        """
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Concrete convenience method: fit(X) then transform(X)."""
        self.fit(X)
        return self.transform(X)

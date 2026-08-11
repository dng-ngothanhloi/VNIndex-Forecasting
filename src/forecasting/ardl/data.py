from __future__ import annotations

from .common import load_inputs


# --- from step_03_load_data.py ---
def run_load_data(context: dict) -> dict:
    data = load_inputs(context["PROJECT_ROOT"])
    context.update(data)

    print("ARDL step 3: data loaded")
    print("  train_df:", context["train_df"].shape)
    print("  val_df  :", context["val_df"].shape)
    print("  test_df :", context["test_df"].shape)
    print("  features:", len(context["pc_cols"]))
    return context


# --- from step_04_validate_pca.py ---
def run_validate_pca(context: dict) -> dict:
    """Validate representation features (PCA or raw) are suitable for ARDL."""
    pc_cols  = context["pc_cols"]
    train_df = context["train_df"]

    k = len(pc_cols)
    n = len(train_df)

    # Feature columns must exist and be numeric
    is_repr_applied = bool(pc_cols) and all(
        train_df[pc_cols].dtypes.apply(lambda t: t.kind in "if")
    )

    can_use_ardl = is_repr_applied and k >= 1 and n > 30

    print("ARDL step 4: representation check")
    print(f"  Representation features numeric? {is_repr_applied}")
    print(f"  Feature columns detected       k = {k}")
    print(f"  Training observations          n = {n}")
    print(f"  Can ARDL be applied?             {can_use_ardl}")

    if not can_use_ardl:
        if not is_repr_applied:
            raise ValueError("Feature columns missing or non-numeric. Run representation pipeline first.")
        if k < 1:
            raise ValueError("No feature columns found (k=0). Check representation output.")
        if n <= 30:
            raise ValueError(f"Training set too small (n={n}). Need > 30 observations.")

    context["is_pca_applied"] = is_repr_applied
    context["can_use_ardl"]   = can_use_ardl
    return context

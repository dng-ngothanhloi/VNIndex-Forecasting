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
    print("  PCs     :", len(context["pc_cols"]))
    return context


# --- from step_04_validate_pca.py ---
def run_validate_pca(context: dict) -> dict:
    pc_cols  = context["pc_cols"]
    train_df = context["train_df"]

    k = len(pc_cols)
    n = len(train_df)

    # PCA columns must exist and be numeric
    is_pca_applied = bool(pc_cols) and all(
        train_df[pc_cols].dtypes.apply(lambda t: t.kind in "if")
    )

    # Minimum: at least 1 PC and enough obs to fit an AR(1) model
    # Rule of thumb: n_train > max_p + k*max_q + 10 (well within any realistic dataset)
    # We use a conservative lower bound: n > 30 and k >= 1
    can_use_ardl = is_pca_applied and k >= 1 and n > 30

    print("ARDL step 4: PCA check")
    print(f"  Is PCA applied?                  {is_pca_applied}")
    print(f"  Principal components detected  k = {k}")
    print(f"  Training observations          n = {n}")
    print(f"  Can ARDL be applied?             {can_use_ardl}")

    if not can_use_ardl:
        if not is_pca_applied:
            raise ValueError("PCA columns missing or non-numeric. Run PCA pipeline first.")
        if k < 1:
            raise ValueError("No principal components found (k=0). Check PCA output.")
        if n <= 30:
            raise ValueError(f"Training set too small (n={n}). Need > 30 observations.")

    context["is_pca_applied"] = is_pca_applied
    context["can_use_ardl"]   = can_use_ardl
    return context

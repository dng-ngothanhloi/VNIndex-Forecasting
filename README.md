# Application of Dimensionality Reduction and Deep Learning in VN-INDEX Forecasting

Dự án áp dụng **PCA** (Principal Component Analysis), **ARDL** (Autoregressive Distributed Lag) và **LSTM** (Long Short-Term Memory) để dự báo chỉ số VN-Index từ dữ liệu giá cổ phiếu niêm yết.

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Python | 3.11+ |
| macOS (Apple Silicon) | Sequoia 15+ / macOS 26+ |
| RAM tối thiểu | 8 GB |

> **Lưu ý TensorFlow trên Apple Silicon:** `tensorflow==2.21` kết hợp `tensorflow-metal==1.2` có lỗi `dlopen libmetal_plugin.dylib` trên macOS 26.x do RPATH không resolve đúng `_pywrap_tensorflow_internal.so` trong môi trường venv. Script setup tự động dùng `tensorflow-macos==2.16.2` thay thế (CPU-safe, ổn định).

---

## Cấu trúc dự án

```
VNIndex-Forecasting/
├── run_pipeline.py                  # Backward-compatible shim → experiments/run_experiment.py
├── configs/
│   └── config.yaml                  # Cấu hình chung (paths, split, PCA, ARDL, LSTM)
├── data/
│   ├── raw/
│   │   └── Data_VNINDEX.csv         # Dữ liệu đầu vào (bắt buộc)
│   └── processed/                   # Tự động sinh ra sau khi chạy pipeline
│       ├── core/                    # vnindex_target.csv, cleaned_data.csv
│       ├── quality/                 # Data quality reports
│       ├── splits/                  # train/val/test_scaled.csv (Z-score)
│       ├── stationarity/            # ADF/KPSS test results
│       └── pca/                     # train/val/test_pca.csv, loadings, metrics
├── experiments/                     # Canonical experiment entry points
│   ├── run_experiment.py            # Unified orchestrator (--full-sweep, --include-multiseed)
│   ├── run_baselines.py             # Persistence + AR(1) baselines (--run-dir)
│   ├── run_ardl_experiment.py       # ARDL standalone experiment
│   ├── run_lstm_experiment.py       # LSTM standalone experiment
│   └── run_lstm_multiseed_stability.py  # Multi-seed stability (seeds 42/52/62/72/82)
├── src/                             # Production source code
│   ├── run_all.py                   # Preprocess + PCA (single CEV)
│   ├── run_all_multi_cev.py         # Multi-CEV orchestrator
│   ├── preprocess.py                # Preprocessing pipeline
│   ├── pca_model.py                 # PCA pipeline (fit Train-only, transform all)
│   ├── utils.py                     # split_by_time, helpers
│   ├── preprocess_steps/            # Step-by-step preprocessing modules
│   ├── preprocessing/               # Preprocessing abstractions
│   ├── reduction/                   # BaseReducer, PCAReducer, NoReduction
│   │   ├── base.py                  # BaseReducer ABC, NotFittedError
│   │   ├── pca.py                   # PCAReducer (wraps sklearn PCA)
│   │   └── noreduction.py           # NoReduction (identity transform baseline)
│   ├── forecasting/                 # Forecasting models
│   │   ├── base.py                  # BaseForecaster ABC
│   │   ├── prediction_result.py     # PredictionResult dataclass
│   │   ├── ar.py                    # Persistence + AR(1) baselines
│   │   ├── ardl_forecaster.py       # ARDLForecaster (BaseForecaster adapter)
│   │   ├── lstm_forecaster.py       # LSTMForecaster (BaseForecaster adapter)
│   │   ├── ardl/                    # ARDL pipeline modules
│   │   │   ├── setup.py             # Environment + path setup
│   │   │   ├── data.py              # Load PCA + VNINDEX, validate
│   │   │   ├── sweep.py             # (P,Q) grid search, model selection
│   │   │   ├── forecast.py          # Final refit on Train+Val, rolling T+1 forecast
│   │   │   ├── export.py            # Export model pickle
│   │   │   ├── reporting.py         # Summary report + figures
│   │   │   ├── diagnostics.py       # ADF stationarity test
│   │   │   └── common.py            # rolling_one_step_forecast, metrics helpers
│   │   └── lstm/                    # LSTM pipeline modules
│   │       ├── setup.py             # TF determinism, path config
│   │       ├── data.py              # Load data, scaling, cross-boundary windowing
│   │       ├── sweep.py             # Lookback×Batch sweep + final_refit_and_forecast
│   │       ├── export.py            # Export model pickle bundle
│   │       └── reporting.py         # Summary report + figures
│   └── evaluation/                  # Model comparison
│       ├── dm_test.py               # Diebold-Mariano test (HAC-corrected)
│       ├── run_dm_test.py           # DM test runner (ARDL vs LSTM)
│       ├── metrics.py               # Shared regression metrics
│       ├── compare_representations.py  # Representation comparison (--run-dir)
│       └── compare_cev_levels.py    # (deprecated) legacy cross-CEV comparison
├── tests/                           # pytest test suite
│   ├── conftest.py                  # Path setup
│   ├── forecasting/                 # Forecasting tests
│   │   ├── test_scientific_correction.py   # P0-1..P0-5 governance tests
│   │   ├── test_report_consistency.py      # Report labeling consistency
│   │   ├── test_ardl_forecaster.py         # ARDLForecaster unit + parity
│   │   ├── test_lstm_forecaster.py         # LSTMForecaster unit tests
│   │   └── test_lstm_multiseed_stability.py
│   ├── preprocessing/               # Preprocessing tests
│   └── reduction/                   # Reduction/PCA tests
├── models/                          # Model artifacts (.pkl)
├── logs/figures/                    # Biểu đồ output (ardl/, lstm/)
├── outputs/                         # Live run outputs (sweep results, forecasts, DM)
│   ├── ardl_vnindex_pca_sweep/      # ARDL sweep_results.csv
│   ├── ardl_vnindex_forecast/       # ARDL forecast CSVs + model pkl
│   ├── lstm_vnindex_sweep/          # LSTM sweep_summary.csv + predictions + pkl
│   ├── model_comparison/            # DM test report + results
│   └── cev_comparison/              # Multi-CEV comparison (when enabled)
├── artifacts/                       # Immutable run snapshots (Run_YYYYMMDD_HHMMSS/)
│                                    #   → xem "Artifact structure" bên dưới
├── docs/                            # Additional documentation
├── requirements.txt
└── local_setup_env.sh               # Script setup môi trường (uv-based)
```

---

## Cài đặt

### Bước 1 – Clone repository

```bash
git clone <repository_url>
cd VNIndex-Forecasting
```

### Bước 2 – Chuẩn bị dữ liệu đầu vào

```bash
mkdir -p data/raw data/processed
# Copy file dữ liệu vào đúng vị trí:
cp /path/to/Data_VNINDEX.csv data/raw/
```

File `Data_VNINDEX.csv` phải chứa các cột: `Ngày`, `Symbol`, `Lần cuối`, `Mở`, `Cao`, `Thấp`, `KL`, `% Thay đổi`.

### Bước 3 – Setup môi trường (khuyến nghị dùng script)

```bash
bash local_setup_env.sh
source .venv/bin/activate
```

Script tự động:
- Tạo virtual environment `.venv`
- Cài toàn bộ dependencies từ `requirements.txt`
- Thay thế `tensorflow==2.21` bằng `tensorflow-macos==2.16.2` (fix lỗi Metal Plugin trên macOS arm64)
- Kiểm tra import tất cả packages

**Hoặc cài thủ công:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Fix TensorFlow cho Apple Silicon:
pip uninstall -y tensorflow tensorflow-metal
pip install tensorflow-macos==2.16.2
```

---

## Chạy experiments

> Đảm bảo đã `source .venv/bin/activate` và đang ở thư mục gốc của project.

Mỗi lần chạy đều tạo một `artifacts/Run_<YYYYMMDD_HHMMSS>/` bất biến chứa
snapshot đầy đủ (data, models, outputs, logs, manifests). Thư mục `outputs/` ở
gốc project là **mutable working dir** — bị ghi đè giữa các lần chạy, không dùng
để so sánh kết quả.

Một lần chạy cho một representation gồm 5 bước:

1. **Preprocess + Representation** – PCA (theo CEV) hoặc NoReduction
2. **ARDL** – sweep (P,Q) → BIC selection → final refit Train+Val → rolling T+1 Test
3. **LSTM** – sweep lookback×batch → Val_RMSE selection → final refit → Test
4. **Diebold-Mariano test** – ARDL vs LSTM trên cùng Test population
5. **Multi-seed stability** (optional) – seeds [42, 52, 62, 72, 82]

---

## Pipeline đầy đủ (khuyến nghị) — 3 lệnh, 1 experiment batch

Đây là workflow chuẩn để tạo **toàn bộ** kết quả nghiên cứu trong **một** thư mục
`artifacts/Run_<timestamp>/` duy nhất, tự khép kín, không phụ thuộc `outputs/` (mutable).

```bash
# ── STEP 1: Full sweep + multi-seed (tất cả representations) ──────────────
python experiments/run_experiment.py --full-sweep --include-multiseed

# → Ghi nhớ Run_<timestamp> in ra ở cuối. Ví dụ: Run_20260811_134344

# ── STEP 2: Baselines (Persistence + AR(1)) ───────────────────────────────
python experiments/run_baselines.py --run-dir artifacts/Run_<timestamp>

# ── STEP 3: Bảng so sánh representations ──────────────────────────────────
python -m src.evaluation.compare_representations --run-dir artifacts/Run_<timestamp>
```

### Step 1 làm gì

`--full-sweep` tạo **một** parent `Run_*` rồi chạy tuần tự 6 child labels:

| Child label | Representation |
|---|---|
| `no_dr` | NoReduction (318 raw scaled features) |
| `pca_cev_0.75` | PCA k=2 |
| `pca_cev_0.80` | PCA k=3 |
| `pca_cev_0.85` | PCA k=4 |
| `pca_cev_0.90` | PCA k=6 |
| `pca_cev_0.95` | PCA k=11 |

Mỗi child chạy đầy đủ, độc lập:

1. **Preprocess + Representation** (PCA fit riêng cho từng CEV, hoặc NoReduction)
2. **ARDL** – sweep (P,Q) → BIC selection → final refit Train+Val → rolling T+1 Test
3. **LSTM** – sweep lookback×batch → Val_RMSE selection → final refit → Test
   (kèm epoch-level tuning history cho **mọi** candidate)
4. **DM test** – ARDL vs LSTM trên cùng 167 Test dates
5. **Multi-seed stability** (chỉ khi có `--include-multiseed`) – seeds [42, 52, 62, 72, 82]

Trước mỗi child, các mutable output dirs được **xoá sạch** để không lẫn dữ liệu cũ;
sau khi chạy xong, toàn bộ được snapshot vào `Run_*/<label>/`.

> **`--include-multiseed`** đảm bảo kết quả multi-seed nằm đúng trong
> `Run_*/<label>/outputs/lstm_vnindex_multiseed/` chứ không rơi ra `outputs/` chung.
> Nếu multi-seed lỗi, child vẫn được đánh dấu `OK` (non-fatal) vì ARDL/LSTM/DM vẫn hợp lệ.

### Artifact structure sau Step 1–3

```
artifacts/Run_20260823_130205/
├── sweep_manifest.json                     # planned/completed/failed labels, multiseed_included
│
├── no_dr/                                  # ─── child label (self-contained) ───
│   ├── run_manifest.json                   #   parent_run_id, label, representation, status
│   ├── config/effective_config.yaml        #   config thực tế dùng cho child này
│   ├── results/run_summary.json            #   ARDL/LSTM/DM/multiseed metrics
│   ├── data/processed/                     #   splits + representation data
│   ├── models/
│   ├── logs/figures/
│   └── outputs/
│       ├── ardl_vnindex_pca_sweep/         #   sweep_results.csv
│       ├── ardl_vnindex_forecast/          #   chapter4_ardl_forecast.csv
│       ├── ardl_vnindex_report/
│       ├── lstm_vnindex_sweep/
│       │   ├── sweep_summary.csv
│       │   ├── predictions_lookback_*.csv
│       │   ├── tuning_history/             #   epoch-level learning curves (mọi candidate)
│       │   └── selected_tuning_history.csv
│       ├── lstm_vnindex_multiseed/         #   per-seed predictions + mean±std
│       └── model_comparison/               #   dm_test_results.csv, dm_test_report.txt
│
├── pca_cev_0.75/  ... pca_cev_0.95/        # cùng structure như trên
│
├── baselines/                              # ─── Step 2 ───
│   ├── persistence/{predictions_val,predictions_test}.csv, summary.json
│   └── ar1/{predictions_val,predictions_test}.csv, summary.json, model_summary.json
│
└── comparison/                             # ─── Step 2 + Step 3 ───
    ├── representation_comparison.{csv,json}    # ARDL vs LSTM per representation
    ├── baseline_comparison.csv                 # Persistence/AR(1)/PCA-ARDL
    ├── pca_ardl_incremental_value.csv          # gain vs Persistence & AR(1)
    ├── baseline_dm_comparison.csv              # DM: baseline vs PCA-ARDL
    └── baseline_interpretation.json
```

### Thời gian ước tính

| Lệnh | Thời gian |
|---|---|
| `--full-sweep` | ~40–60 phút |
| `--full-sweep --include-multiseed` | ~70–120 phút |
| `run_baselines.py` | < 1 giây |
| `compare_representations` | < 1 giây |

---

## Chạy từng phần (debug / chạy lại một representation)

### Single representation (vẫn dùng chung schema `Run_*/<label>/`)

```bash
python experiments/run_experiment.py --reduction none
python experiments/run_experiment.py --reduction pca --cev 0.75
python experiments/run_experiment.py --reduction pca --cev 0.85

# Kèm multi-seed cho representation đó:
python experiments/run_experiment.py --reduction pca --cev 0.85 --include-multiseed
```

Mỗi lệnh tạo `artifacts/Run_<ts>/` mới với **một** child label bên trong.

### Tuỳ chọn bổ sung

```bash
python experiments/run_experiment.py --skip-preprocess   # Rerun ARDL + LSTM (dữ liệu đã có)
python experiments/run_experiment.py --skip-ardl         # Bỏ qua ARDL
python experiments/run_experiment.py --skip-lstm         # Bỏ qua LSTM
python experiments/run_experiment.py --skip-dm           # Bỏ qua DM test
python experiments/run_experiment.py --continue-on-error # Không dừng nếu 1 step fail
```

### Multi-seed LSTM stability standalone

```bash
python experiments/run_lstm_multiseed_stability.py
python experiments/run_lstm_multiseed_stability.py --skip-dm              # bỏ DM per-seed
python experiments/run_lstm_multiseed_stability.py --validation-diagnostic # per-seed own best_epoch
```

**Yêu cầu:** `data/processed/pca/` phải đang chứa representation muốn test.

**Cách hoạt động:**
1. Chạy LSTM tuning sweep với seed=42 (reference) → freeze (lookback, batch_size, best_epoch)
2. Mỗi seed trong [42, 52, 62, 72, 82]: NEW model instance, refit Train+Val với cùng
   hyperparameters đã freeze, forecast Test
3. Report mean ± std cho Test RMSE/MAE/MAPE/R²

> Chạy standalone sẽ ghi ra `outputs/lstm_vnindex_multiseed/` (mutable).
> Muốn kết quả nằm trong `artifacts/`, dùng `--include-multiseed` ở Step 1.

**`--validation-diagnostic`** (chẩn đoán, không refit/không Test): với cùng LB/BS đã chọn,
mỗi seed chạy Train→Val với EarlyStopping riêng để lấy `own_best_epoch`. Dùng để phân biệt
hiện tượng `best_epoch=1` là do seed hay do distribution shift.

---

## Chạy từng module riêng lẻ (advanced)

Dùng khi cần debug sâu một module cụ thể.

**Bước 1: Tiền xử lý + PCA**

```bash
python src/run_all.py --config configs/config.yaml
```

Output chính:
- `data/processed/core/` – cleaned data, vnindex_target.csv
- `data/processed/splits/` – train/val/test_scaled.csv (Z-score, fit Train-only)
- `data/processed/pca/` – train/val/test_pca.csv, pca_loadings.csv, pca_threshold_summary.csv
- `models/pca_model.pkl`, `models/scaler_params.pkl`

**Bước 2: ARDL**

```bash
python experiments/run_ardl_experiment.py
```

Yêu cầu: Bước 1 đã hoàn thành (`data/processed/pca/*.csv` phải tồn tại).

Thực hiện:
- Sweep tất cả (P,Q) ∈ p_values × q_values, fit Train-only, rolling one-step forecast Val
- Auto-select theo BIC (với RMSE_val guard)
- Final refit trên Train+Val, rolling T+1 forecast toàn bộ Test
- Export model pkl + forecast CSV

**Bước 3: LSTM**

```bash
python experiments/run_lstm_experiment.py
```

Yêu cầu: Bước 1 đã hoàn thành.

Thực hiện:
- Sweep tất cả lookback × batch_size, fit Train with EarlyStopping, evaluate Val (cross-boundary windowing)
- Select lowest Val_RMSE
- Final refit trên Train+Val (best_epoch epochs, no EarlyStopping), forecast Test (cross-boundary)
- Export model pkl + predictions CSV + summary report + figures

**Bước 4: Diebold-Mariano Test (so sánh ARDL vs LSTM)**

```bash
python -m src.evaluation.run_dm_test
```

Yêu cầu: Bước 2 và 3 đã hoàn thành (cần forecast CSVs cho cả hai model).

Bước 2 và 3 **không phụ thuộc nhau**, có thể chạy song song sau khi Bước 1 hoàn thành.

**Bước 5: Baselines (Persistence + AR(1))**

```bash
python experiments/run_baselines.py --run-dir artifacts/Run_<timestamp>
```

Đọc VNINDEX + ARDL forecasts từ `Run_*` đã có, tính persistence/AR(1), so sánh + DM test.
Không train model nào (chỉ 2 lần OLS fit nhỏ).

---

## Cấu hình quan trọng (`configs/config.yaml`)

### Data Split Strategy

```yaml
preprocess:
  train_ratio: 0.65
  val_ratio: 0.15               # Disjoint chronological split
  use_overlap_val: false        # false = non-overlap (Train/Val/Test disjoint)
  test_ratio: 0.20              # Informational only (derived as 1 - train - val)
```

Kết quả split cho N=826 observations: Train=536 / Val=123 / Test=167.

### Representation (PCA / NoReduction)

```yaml
reduction:
  method: pca              # pca | none  (none = NoReduction, 318 raw scaled features)

pca:
  explained_variance_threshold: 0.75              # CEV cho single run
  cev_thresholds: [0.75, 0.80, 0.85, 0.90, 0.95]  # Danh sách CEV cho --full-sweep
  use_multi_cev: false                            # (legacy) true = dùng run_all_multi_cev.py
```

`--reduction` và `--cev` trên CLI override 2 key trên. Default giữ nguyên `pca`
để mọi lệnh hiện có tái tạo đúng pipeline PCA cũ.

### ARDL Model Selection

```yaml
ardl:
  p_values: [1, 2, 3, 4, 5]      # AR lags to sweep
  q_values: [1, 2, 3, 4, 5]      # Exog lag depth (PC.L1..PC.Lq; no PC.L0)
  causal: true                    # No contemporaneous exogenous (information boundary)
  hold_back: 5                    # Fixed burn-in for fair IC comparison
  selection_criterion: "BIC"      # BIC | AIC | HQIC | RMSE_val | MAE_val
  bic_thresholds: null            # Upper bound filter (null = disabled)
  rmse_val_thresholds: 17.0       # OOS validation guard
  selected_pair: null             # null = auto-select; [P,Q] = hardcode
  use_ensemble: false             # true = average top-N models
```

### LSTM Hyperparameters

```yaml
lstm:
  lookback_values: [20, 30, 40, 50, 60]
  batch_size_values: [16, 32]
  epochs: 150                    # Max epochs during tuning (with EarlyStopping)
  learning_rate: 0.0001
  early_stopping_patience: 25
  min_epochs: 30                 # Hard floor: refuse early-stop before this
  lstm_units: [64, 32]
  dense_units: [16]
  dropout_rate: 0.2
  export_lookback: null          # null = auto-pick lowest Val_RMSE
  export_batch_size: null
```

---

## Scientific methodology

### ARDL

- **Kiểu mô hình:** Fixed-coefficient rolling one-step-ahead ARDL with PCA exogenous variables
- **Forecast horizon:** T+1 (rolling one-step, actual observed history, NOT recursive substitution)
- **Tuning:** Fit Train-only → rolling one-step Val → BIC selection (P0-3A)
- **Final refit:** Unique Train+Val (no duplicate dates), rolling T+1 forecast trên toàn bộ Test
- **Information boundary:** `causal=True` (no PC.L0), `hold_back=5`

### LSTM

- **Kiểu mô hình:** Sequential LSTM (2 layers + Dense) with PCA + target history features
- **Forecast horizon:** T+1
- **Tuning:** Fit Train (max available per lookback) with EarlyStopping, evaluate fixed 123 Val targets (cross-boundary windowing, P0-4)
- **Selection:** Lowest Val_RMSE across (lookback × batch_size) grid
- **Final refit (D4):** NEW model instance, fit Train+Val for exactly `best_epoch` epochs (no EarlyStopping), forecast ALL Test targets via cross-boundary windowing
- **Multi-seed stability:** Refit same frozen (lookback, batch_size, best_epoch) with seeds [42, 52, 62, 72, 82]

### Baselines (Persistence, AR(1))

Cả hai dùng **cùng** dataset / split / Test dates / T+1 horizon / rolling one-step
information boundary như PCA-ARDL & LSTM (verify fail-fast: same N=167, same dates, same y_true).

- **Persistence:** `y_hat(t) = actual y(t-1)`. Không fit. Target Test đầu tiên dùng
  observation cuối của Val.
- **AR(1):** `y_t = c + φ·y_{t-1} + ε`, OLS.
  - Val diagnostic: fit Train-only (536 obs)
  - Final Test: refit trên unique Train+Val (659 obs), freeze coefficients,
    rolling one-step với **actual** `y(t-1)` (không recursive substitution)

Mục đích: trả lời câu hỏi *PCA-ARDL có giá trị dự báo tăng thêm so với persistence
đơn thuần / mô hình tự hồi quy hay không*.

### Diebold-Mariano Test

- **Fail-fast P0-5:** Requires ARDL and LSTM to predict the SAME Test target dates, SAME y_true, SAME n (167) — refuses to merge mismatched populations
- **Significance reporting:** Distinguishes "observed lower loss" from "statistically significant" — marginal results (0.05 ≤ p < 0.10) are explicitly labeled as NOT confirmed accuracy differences at the 5% level
- **Hai loại so sánh riêng biệt:**
  - `ARDL vs LSTM` trong cùng một representation (`Run_*/<label>/outputs/model_comparison/`)
  - `Baseline vs PCA-ARDL` (`Run_*/comparison/baseline_dm_comparison.csv`)
  - Không dùng DM ARDL-vs-LSTM để kết luận "PCA thắng NoReduction" — đó là so sánh khác

---

## Chạy tests

```bash
# Full test suite
pytest -q

# Chỉ forecasting tests
pytest tests/forecasting -q

# Specific test file
pytest tests/forecasting/test_report_consistency.py -q

# Scientific correction tests
pytest tests/forecasting/test_scientific_correction.py -q
```

---

## Troubleshooting

### TensorFlow lỗi `dlopen libmetal_plugin.dylib` trên macOS

```
NotFoundError: dlopen(libmetal_plugin.dylib): Library not loaded: @rpath/_pywrap_tensorflow_internal.so
```

Giải pháp:

```bash
pip uninstall -y tensorflow tensorflow-metal
pip install tensorflow-macos==2.16.2
```

### ImportError / ModuleNotFoundError

Đảm bảo chạy từ **thư mục gốc** của project:

```bash
cd /path/to/VNIndex-Forecasting
source .venv/bin/activate
python experiments/run_experiment.py
```

### ARDL/LSTM báo thiếu file PCA

Chạy preprocessing + PCA trước:

```bash
python src/run_all.py --config configs/config.yaml
```

---

## Dependencies

| Package | Mục đích |
|---|---|
| numpy | Tính toán mảng |
| pandas | Xử lý dữ liệu |
| scikit-learn | PCA, StandardScaler |
| statsmodels | ARDL, ADF/KPSS test |
| tensorflow-macos | LSTM (Apple Silicon) |
| scipy | DM test, statistics |
| matplotlib | Visualization |
| pyyaml | Config |
| tabulate | ARDL summary tables |
| hypothesis | Property-based testing |

---
**Author:**
Ngo Thanh Loi, MCS Student at Duy Tan University, Vietnam
Email: <ngothanhloi@dtu.edu.vn>
LinkedIn: https://www.linkedin.com/in/ngo-thanh-loi/

**Contribution team:**

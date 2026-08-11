# Application of Dimensionality Reduction and Deep Learning in VN-INDEX Forecasting

Dự án áp dụng **PCA** (Principal Component Analysis), **ARDL** (Autoregressive Distributed Lag) và **LSTM** (Long Short-Term Memory) để dự báo chỉ số VN-Index từ dữ liệu giá cổ phiếu niêm yết.

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Python | 3.11+ |
| macOS (Apple Silicon) | Sequoia 15+ / macOS 26+ |
| RAM tối thiểu | 8 GB |

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
│   ├── run_experiment.py            # Unified orchestrator (preprocess+PCA → ARDL → LSTM → DM)
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
│   ├── reduction/                   # BaseReducer, PCAReducer
│   │   ├── base.py                  # BaseReducer ABC, NotFittedError
│   │   └── pca.py                   # PCAReducer (wraps sklearn PCA)
│   ├── forecasting/                 # Forecasting models
│   │   ├── base.py                  # BaseForecaster ABC
│   │   ├── prediction_result.py     # PredictionResult dataclass
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
│       └── compare_cev_levels.py    # Cross-CEV comparison
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
├── artifacts/                       # Timestamped run snapshots (Run_YYYYMMDD_HHMMSS/)
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

### Cách 1 – Unified Pipeline (khuyến nghị)

Chạy toàn bộ pipeline bằng 1 lệnh duy nhất:

```bash
python run_pipeline.py
# hoặc tương đương:
python experiments/run_experiment.py
```

Lệnh này tự động thực hiện:
1. **Preprocess + PCA** (CEV threshold từ `configs/config.yaml`)
2. **ARDL grid search** – sweep (P,Q), selection by BIC, final refit on Train+Val, rolling T+1 forecast on Test
3. **LSTM sweep** – lookback × batch_size grid, selection by Val_RMSE, final refit (Train+Val, best_epoch epochs, no EarlyStopping), forecast Test via cross-boundary windowing
4. **Diebold-Mariano test** – so sánh thống kê ARDL vs LSTM trên cùng Test population
5. **Artifacts** – lưu toàn bộ kết quả vào `artifacts/Run_<YYYYMMDD_HHMMSS>/`

**Multi-CEV sweep** (so sánh nhiều ngưỡng PCA cùng lúc):

```bash
# Recommended: full experiment sweep (each run isolated in artifacts/)
python experiments/run_experiment.py --full-sweep

# Then compare:
python -m src.evaluation.compare_representations

# Individual runs (still work):
python experiments/run_experiment.py --reduction none

# PCA at various CEV levels:
python experiments/run_experiment.py --reduction pca --cev 0.75
python experiments/run_experiment.py --reduction pca --cev 0.80
python experiments/run_experiment.py --reduction pca --cev 0.85
python experiments/run_experiment.py --reduction pca --cev 0.90
python experiments/run_experiment.py --reduction pca --cev 0.95
# Compare all completed runs:
python -m src.evaluation.compare_representations

```

**Tùy chọn bổ sung:**

```bash
python experiments/run_experiment.py --skip-preprocess   # Rerun ARDL + LSTM (dữ liệu đã có)
python experiments/run_experiment.py --skip-ardl         # Bỏ qua ARDL
python experiments/run_experiment.py --skip-lstm         # Bỏ qua LSTM
python experiments/run_experiment.py --skip-dm           # Bỏ qua DM test
python experiments/run_experiment.py --continue-on-error # Không dừng nếu 1 step fail
python experiments/run_experiment.py --no-collect        # Không copy vào artifacts/
```

**Mmulti-seed LSTM stability:**


```bash
python experiments/run_lstm_multiseed_stability.py
```

**Yêu cầu trước khi chạy:**
- Pipeline preprocess + PCA (hoặc NoReduction) + LSTM tuning sweep phải đã hoàn thành trước (ít nhất `data/processed/pca/` và `outputs/lstm_vnindex_sweep/` phải tồn tại)

**Cách hoạt động:**
1. Chạy LSTM tuning sweep với seed=42 (reference run) — freeze (lookback, batch_size, best_epoch) được chọn
2. Với mỗi seed trong [42, 52, 62, 72, 82]: tạo NEW model instance, refit trên Train+Val với cùng (lookback, batch_size, best_epoch), forecast Test
3. Report: mean ± std cho Test RMSE/MAE/MAPE across 5 seeds

**Options:**
```bash
# Skip DM comparison per seed (chỉ report LSTM stability):
python experiments/run_lstm_multiseed_stability.py --skip-dm
```

**Thứ tự chạy đầy đủ cho một representation:**
```bash
# 1. Single run (tuning + final refit, seed=42):
python experiments/run_experiment.py --reduction pca --cev 0.75

# 2. Multi-seed stability (reuses frozen selection from step 1):
python experiments/run_lstm_multiseed_stability.py
```

Output sẽ lưu vào `outputs/lstm_vnindex_multiseed/` với per-seed predictions và summary statistics.


| Lệnh | Thực hiện | Thời gian |
|---|---|---|
| `python experiments/run_experiment.py` | Full single-CEV (preprocess+PCA+ARDL+LSTM+DM) | ~5-10 phút |
| `python experiments/run_experiment.py --multi-cev` | Trên × N CEV levels + compare | ~15-30 phút |

---

### Cách 2 – Chạy từng bước riêng biệt

Dùng khi cần debug hoặc chạy riêng từng phần.

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

---

### Multi-seed LSTM stability (tùy chọn)

Sau khi Bước 3 hoàn thành (tuning selection frozen), chạy multi-seed stability:

```bash
python experiments/run_lstm_multiseed_stability.py
```

Refit cùng (lookback, batch_size, best_epoch) với seeds [42, 52, 62, 72, 82], report mean ± std.

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

### PCA Threshold

```yaml
pca:
  explained_variance_threshold: 0.75   # Active threshold cho single-CEV run
  use_multi_cev: false                 # true = sweep all cev_thresholds
  cev_thresholds: [0.85, 0.90, 0.95]  # Thresholds cho multi-CEV mode
```

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

### Diebold-Mariano Test

- **Fail-fast P0-5:** Requires ARDL and LSTM to predict the SAME Test target dates, SAME y_true, SAME n (167) — refuses to merge mismatched populations
- **Significance reporting:** Distinguishes "observed lower loss" from "statistically significant" — marginal results (0.05 ≤ p < 0.10) are explicitly labeled as NOT confirmed accuracy differences at the 5% level

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

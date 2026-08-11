#!/usr/bin/env bash
# =============================================================================
# local_setup_env.sh  –  Setup môi trường cho VNIndexPredictor
# Dùng `uv` để tránh lỗi ensurepip/pyexpat của Homebrew Python trên macOS 26+
#
# Chạy từ thư mục gốc project:  bash local_setup_env.sh
# Activate sau khi setup:        source .venv/bin/activate
# =============================================================================
set -euo pipefail

PYTHON_VERSION="3.11"
VENV_DIR=".venv"

echo "============================================================"
echo " VNIndexPredictor – Environment Setup (via uv)"
echo "============================================================"

# ── 1. Kiểm tra uv ───────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[ERROR] uv không tìm thấy."
    echo ""
    echo "Cài uv (chọn một trong hai cách):"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  brew install uv"
    echo ""
    echo "Sau đó mở terminal mới và chạy lại script này."
    exit 1
fi

echo "[INFO]  uv: $(uv --version)"

# ── 2. Tạo virtual environment bằng uv ──────────────────────────
# uv dùng Python build riêng, không phụ thuộc Homebrew Python
if [ -d "$VENV_DIR" ]; then
    echo "[INFO]  Virtual env '$VENV_DIR' đã tồn tại, bỏ qua tạo mới."
    echo "        (Xoá thư mục .venv nếu muốn tạo lại: rm -rf .venv)"
else
    echo "[INFO]  Tạo virtual environment Python $PYTHON_VERSION tại $VENV_DIR ..."
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

# ── 3. Xác nhận Python trong venv ────────────────────────────────
VENV_PYTHON="$VENV_DIR/bin/python"
echo "[INFO]  Python in venv: $($VENV_PYTHON --version)"

# ── 4. Cài core dependencies ─────────────────────────────────────
echo ""
echo "[INFO]  Cài đặt dependencies từ requirements.txt ..."
uv pip install -r requirements.txt --python "$VENV_PYTHON"
echo "[OK]    Core dependencies installed."

# ── 5. Fix TensorFlow cho Apple Silicon (macOS arm64) ────────────
# tensorflow==2.16.2 trong requirements.txt là CPU-safe build.
# KHÔNG cài tensorflow-metal để tránh lỗi dlopen libmetal_plugin.dylib.
# Bỏ comment dòng dưới chỉ khi muốn thử GPU Metal (có thể không ổn định):
# uv pip install tensorflow-metal==1.1.0 --python "$VENV_PYTHON"

# ── 6. Kiểm tra import nhanh ─────────────────────────────────────
echo ""
echo "[CHECK] Kiểm tra import các thư viện chính ..."
"$VENV_PYTHON" - <<'PYEOF'
import sys

checks = {
    "numpy":       "numpy",
    "pandas":      "pandas",
    "scipy":       "scipy",
    "sklearn":     "scikit-learn",
    "statsmodels": "statsmodels",
    "tensorflow":  "tensorflow",
    "matplotlib":  "matplotlib",
    "yaml":        "pyyaml",
    "joblib":      "joblib",
    "tabulate":    "tabulate",
}

failed = []
for mod, label in checks.items():
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"  OK  {label}: {v}")
    except Exception as e:
        short = str(e).split('\n')[0][:80]
        print(f"  FAIL  {label}: {short}")
        failed.append(label)

print()
if failed:
    print(f"[WARN]  {len(failed)} package(s) không import được: {failed}")
    sys.exit(1)
else:
    print("[OK]  Tất cả packages import thành công.")
PYEOF

# ── 7. Chuẩn bị thư mục data ─────────────────────────────────────
echo ""
echo "[INFO]  Kiểm tra cấu trúc thư mục data/ ..."
for d in data/raw data/processed; do
    if [ ! -d "$d" ]; then
        mkdir -p "$d"
        echo "[MKDIR] $d"
    else
        echo "[OK]    $d đã tồn tại"
    fi
done

if [ ! -f "data/raw/Data_VNINDEX.csv" ]; then
    echo "[WARN]  data/raw/Data_VNINDEX.csv chưa có."
    echo "        Hãy copy file dữ liệu vào thư mục data/raw/ trước khi chạy pipeline."
else
    echo "[OK]    data/raw/Data_VNINDEX.csv found"
fi

# ── 8. Done ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup hoàn tất!"
echo "============================================================"
echo ""
echo "Kích hoạt môi trường:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Chạy pipeline (sau khi activate):"
echo "  python src/run_all.py --config config/config.yaml   # Preprocess + PCA"
echo "  cd ardl && python run_all_ardl.py                   # ARDL model"
echo "  cd lstm && python run_all_lstm.py                   # LSTM model"
echo ""

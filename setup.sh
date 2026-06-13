#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh  –  One-shot environment + project setup
# Usage: bash setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -e

echo "============================================================"
echo "  Hand Gesture Recognition – Setup"
echo "============================================================"

# 1. Python check
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Please install Python 3.9+."
  exit 1
fi
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VER"

# 2. Virtual environment
if [ ! -d ".venv" ]; then
  echo ""
  echo "Creating virtual environment (.venv) …"
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "Virtual environment activated."

# 3. Upgrade pip + install deps
echo ""
echo "Installing dependencies …"
pip install --upgrade pip --quiet
pip install -r requirements.txt

# 4. Kaggle credentials reminder
echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  NEXT STEPS:"
echo "──────────────────────────────────────────────────────────────"
echo ""
echo "  1) Set up Kaggle credentials (needed for dataset download):"
echo "       https://www.kaggle.com/settings → API → Create New Token"
echo "       Place kaggle.json at ~/.kaggle/kaggle.json"
echo "       chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "  2) Download the dataset:"
echo "       python download_dataset.py"
echo ""
echo "  3) Train the model (30-60 min on CPU, ~10 min on GPU):"
echo "       python src/train.py"
echo ""
echo "  4) (Optional) Evaluate the model:"
echo "       python src/evaluate.py"
echo ""
echo "  5) Run the Streamlit app:"
echo "       streamlit run app.py"
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"

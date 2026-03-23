"""
run_all_millets.py
==================
Master automation script for Agrowcart AI Millet Expansion.
Runs the full pipeline (Fetch → Synthesize → Train → Export) for all 7 Indian millets.
Automatically skips millets that already have a trained model in models/
"""
import subprocess
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# All 7 millets to process (in order of data availability)
ALL_MILLETS = ['Bajra', 'Jowar', 'Ragi', 'Kodo', 'Foxtail', 'Barnyard', 'Little']

def run_step(label, cmd_args):
    """Run a subprocess step, print output, and return success bool."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print('='*55)
    result = subprocess.run(
        [sys.executable] + cmd_args,
        cwd=BASE_DIR,
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f"⚠️  Step failed (exit {result.returncode}) — skipping to next millet.")
        return False
    return True

def model_exists(millet):
    path = os.path.join(BASE_DIR, "models", f"{millet.lower()}_lstm_model.pth")
    return os.path.exists(path)

def raw_data_exists(millet):
    path = os.path.join(BASE_DIR, "datasets", f"{millet.lower()}_massive_dataset.csv")
    return os.path.exists(path)

def run_millet_pipeline(millet):
    m = millet.lower()

    # --- STEP 1: Fetch Live Data ---
    if raw_data_exists(millet):
        print(f"⏭️  [{millet}] Raw data already exists, skipping fetch.")
    else:
        ok = run_step(
            f"[{millet}] STEP 1: Fetching live Agmarknet data",
            [os.path.join("src", "fetch_millet_data.py"), millet]
        )
        if not ok:
            return False

    # --- STEP 2: Synthesize Training Dataset ---
    processed_path = os.path.join(BASE_DIR, "datasets", f"{m}_training_processed.csv")
    if os.path.exists(processed_path):
        print(f"⏭️  [{millet}] Training dataset already exists, skipping synthesis.")
    else:
        ok = run_step(
            f"[{millet}] STEP 2: Synthesizing training dataset",
            [os.path.join("src", "build_training_dataset.py"), millet]
        )
        if not ok:
            return False

    # --- STEP 3: Train LSTM Model ---
    if model_exists(millet):
        print(f"⏭️  [{millet}] Model already exists, skipping training.")
    else:
        ok = run_step(
            f"[{millet}] STEP 3: Training LSTM model (this takes ~5 min)",
            [
                os.path.join("src", "train_models.py"),
                f"datasets/{m}_training_processed.csv",
                m
            ]
        )
        if not ok:
            return False

    print(f"\n✅ [{millet}] PIPELINE COMPLETE! Model ready in models/{m}_lstm_model.pth")
    return True

if __name__ == "__main__":
    print("\n🌾 AGROWCART AI — FULL MILLET EXPANSION 🌾")
    print("Covering all 7 major Indian millets...\n")

    results = {}
    for millet in ALL_MILLETS:
        success = run_millet_pipeline(millet)
        results[millet] = "✅ Done" if success else "❌ Failed"

    # --- FINAL REPORT ---
    print("\n" + "="*55)
    print("   FINAL MILLET EXPANSION REPORT")
    print("="*55)
    for millet, status in results.items():
        print(f"  {status}  {millet}")

    print("\n🚀 All models are now ready in models/")
    print("   Restart api_server.py to serve all millets!\n")

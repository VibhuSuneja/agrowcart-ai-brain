import os
import subprocess
import time
from datetime import datetime

def run_step(command, description):
    print(f"--- 🚀 Executing: {description} ---")
    try:
        result = subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return False

def autopilot_main():
    start_time = time.time()
    print(f"🌟 Starting AI Brain Autopilot Refresh Cycle ({datetime.now()})")
    
    # Step 1: Clean old training data to save space on Render/GitHub
    print("🧹 Cleaning environment...")
    if os.path.exists("datasets"):
        for f in os.listdir("datasets"):
            if f.endswith("_massive_dataset.csv"): # Delete raw API dumps, keep processed ones
                os.remove(os.path.join("datasets", f))

    # Step 2: Run the full Master Pipeline
    # This fetches data, synthesizes patterns, and trains the LSTM models
    success = run_step("python run_all_millets.py", "Full Millet Master Pipeline")
    if not success: return

    # Step 3: Verify Accuracy & Generate Health Report
    success = run_step("python src/check_accuracy.py", "AI Confidence Verification")
    if not success: return

    # Step 4: Commit and Push New Brain Weights to GitHub
    # This automatically triggers a redeploy on Render
    print("📦 Syncing new AI weights to cloud...")
    run_step("git add models/*.pth models/*.gz models/health_report.csv", "Staging new models")
    run_step('git commit -m "🤖 [Autopilot] Weekly AI Brain Refresh - Patterns Updated"', "Committing changes")
    run_step("git push origin main", "Pushing to GitHub")

    duration = (time.time() - start_time) / 60
    print(f"✅ Autopilot Cycle Complete! Total time: {duration:.2f} minutes")
    print("Farmer project is now 100% up-to-date with freshest mandi patterns.")

if __name__ == "__main__":
    autopilot_main()

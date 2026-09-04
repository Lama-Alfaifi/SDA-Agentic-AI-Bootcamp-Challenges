import os
import sys
import time
import subprocess
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
STREAMLIT_PORT = os.getenv("STREAMLIT_PORT", "8501")

def start_services():
    print("=" * 65)
    print("🚀 Starting National Transformation Program RAG System")
    print("=" * 65)
    print(f"📡 FastAPI Backend:   http://localhost:{API_PORT}")
    print(f"🎨 Streamlit UI:       http://localhost:{STREAMLIT_PORT}")
    print("=" * 65)

    python_exe = sys.executable

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # 1. Start FastAPI backend
    api_cmd = [
        python_exe, "-m", "uvicorn", "api:app",
        "--host", API_HOST,
        "--port", str(API_PORT)
    ]
    print("[1/2] Launching FastAPI Backend Server...")
    api_process = subprocess.Popen(api_cmd, env=env)

    # Give FastAPI a moment to initialize
    time.sleep(2)

    # 2. Start Streamlit frontend
    st_cmd = [
        python_exe, "-m", "streamlit", "run", "app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    print("[2/2] Launching Streamlit Web Interface...")
    st_process = subprocess.Popen(st_cmd, env=env)

    print("\n✅ Both services are running! Press Ctrl+C to stop.\n")

    try:
        api_process.wait()
        st_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        api_process.terminate()
        st_process.terminate()
        api_process.wait()
        st_process.wait()
        print("Done.")

if __name__ == "__main__":
    start_services()

import os
import sys
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def install_dependencies():
    """Ensure all required packages are installed."""
    required_packages = ["fastapi", "uvicorn", "pydantic", "python-dotenv", "google-generativeai"]
    missing_packages = []
    
    for package in required_packages:
        pkg_name = "dotenv" if package == "python-dotenv" else ("google.generativeai" if package == "google-generativeai" else package)
        try:
            __import__(pkg_name)
        except ImportError:
            # Match package name for install
            missing_packages.append(package)
            
    if missing_packages:
        print(f"[*] Missing dependencies detected: {missing_packages}")
        print("[*] Installing missing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("[+] Successfully installed dependencies!")
        except Exception as e:
            print(f"[-] Error installing dependencies: {e}")
            print("[-] Please run 'pip install -r requirements.txt' manually.")
            sys.exit(1)

if __name__ == "__main__":
    # Install dependencies automatically if any are missing
    install_dependencies()
    
    # Import uvicorn dynamically after potential installation
    import uvicorn
    
    # Run server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    
    print(f"\n========================================================")
    print(f"   LAUNCHING DRUG INTERACTION CHECKER MULTI-AGENT APP")
    print(f"========================================================")
    print(f"[*] Host: {host}")
    print(f"[*] Port: {port}")
    print(f"[*] URL: http://{host}:{port}")
    print(f"========================================================\n")
    
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)

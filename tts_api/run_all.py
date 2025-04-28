import os
import subprocess
import time
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_application():
    """Run the FastAPI application"""
    print("Starting the application...")
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", "8000"))
    
    # Run the application
    subprocess.Popen([sys.executable, "run.py"])

def run_ngrok():
    """Run ngrok"""
    print("Starting ngrok...")
    subprocess.Popen([sys.executable, "start_ngrok.py"])

if __name__ == "__main__":
    # Run the application
    run_application()
    
    # Wait for the application to start
    time.sleep(3)
    
    # Run ngrok
    run_ngrok()
    
    print("Application and ngrok are running. Press Ctrl+C to stop.")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping application and ngrok...")
        # Find and kill the processes
        subprocess.run(["pkill", "-f", "run.py"])
        subprocess.run(["pkill", "-f", "start_ngrok.py"])
        print("Application and ngrok stopped.") 
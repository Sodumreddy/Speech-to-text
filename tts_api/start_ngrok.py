import os
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def start_ngrok():
    # Get ngrok authtoken from environment variable or use the one in ngrok.yml
    authtoken = os.getenv("NGROK_AUTHTOKEN", "2syLTH4JniIBTNGjGSqJ6Vt52cQ_7vpToaxLr7wBLowK56D5n")
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", "8000"))
    
    # Start ngrok
    print(f"Starting ngrok on port {port}...")
    subprocess.Popen(["ngrok", "http", str(port), "--authtoken", authtoken])
    
    # Wait for ngrok to start
    time.sleep(3)
    
    # Get the ngrok URL
    try:
        import requests
        response = requests.get("http://localhost:4040/api/tunnels")
        tunnels = response.json()["tunnels"]
        for tunnel in tunnels:
            if tunnel["proto"] == "https":
                print(f"Ngrok URL: {tunnel['public_url']}")
                print(f"Use this URL in your Twilio webhook configuration: {tunnel['public_url']}/voice")
                break
    except Exception as e:
        print(f"Error getting ngrok URL: {e}")
        print("Please check the ngrok web interface at http://localhost:4040")

if __name__ == "__main__":
    start_ngrok() 
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse

app = FastAPI()

# Optional: Allow frontend apps to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === /voice endpoint for Twilio Call Handling ===
@app.post("/voice")
async def voice(request: Request):
    response = VoiceResponse()
    
    response.say("Welcome to the transcription service. Connecting your call now.")
    
    # Twilio starts streaming audio to our WebSocket
    response.start().stream(url="wss://e6f3-50-96-155-176.ngrok-free.app/ws/transcription")
    
    # Dial the recipient's phone number
    response.dial("17633369510")  # Replace with your destination number

    return Response(content=str(response), media_type="application/xml")

# === WebSocket Endpoint for Streaming Audio Data ===
@app.websocket("/ws/transcription")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket Connected]")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"[WebSocket Received]: {data}")

    except WebSocketDisconnect:
        print("[WebSocket Disconnected]")

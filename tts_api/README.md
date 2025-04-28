# Speech-to-Text with Twilio Integration

This application provides speech-to-text functionality with speaker diarization and keyword extraction. It also integrates with Twilio for phone call transcription.

## Features

- Record audio from microphone and transcribe it
- Speaker diarization (identifying different speakers)
- Keyword extraction from transcriptions
- Twilio integration for phone call transcription
- Call history with transcriptions

## Prerequisites

- Python 3.8+
- MongoDB
- ngrok (for Twilio webhook)
- Twilio account (for phone call transcription)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your configuration (see `.env.example`)

## Configuration

Create a `.env` file with the following variables:

```
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
DB_NAME=Speech-to-text
COLLECTION_NAME=voice

# API Keys
ASSEMBLYAI_API_KEY=your_assemblyai_api_key

# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number

# Application Settings
WHISPER_MODEL=base
CHUNK=1024
FORMAT=16
CHANNELS=1
RATE=16000
RECORD_SECONDS=30

# Output Directories
OUTPUT_DIR=recorded_audio
UPLOAD_DIR=uploads

# ngrok Configuration
NGROK_AUTHTOKEN=your_ngrok_auth_token
PORT=8000
```

## Running the Application

1. Start MongoDB:
   ```
   mongod
   ```

2. Start the application:
   ```
   python run.py
   ```

3. Start ngrok (in a separate terminal):
   ```
   python start_ngrok.py
   ```

4. Configure Twilio webhook:
   - Go to your Twilio console
   - Set the voice webhook URL to your ngrok URL + `/voice`
   - Example: `https://your-ngrok-url.ngrok-free.app/voice`

5. Access the application:
   - Open your browser and go to `http://localhost:8000/static/index.html`

## Usage

### Microphone Recording

1. Select your microphone from the dropdown
2. Click "Start Recording"
3. Speak into your microphone
4. Click "Stop Recording" when done
5. View the transcription result

### Phone Call Transcription

1. Call your Twilio phone number
2. The call will be transcribed and stored in the database
3. View the transcription in the "Call History" section

## Troubleshooting

- If MongoDB connection fails, make sure MongoDB is running
- If AssemblyAI transcription fails, check your API key
- If Twilio integration fails, check your Twilio credentials and webhook configuration
- If ngrok fails, check your authtoken and make sure ngrok is installed 
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import os
import sys
import time
import logging
import threading
import wave
import pyaudio
import whisper
import requests
import pymongo
import numpy as np
import torch
import uvicorn
from datetime import datetime
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import shutil
import uuid
from bson import ObjectId
from bson.json_util import dumps, loads
from werkzeug.utils import secure_filename
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "Speech-to-text"
COLLECTION_NAME = "voice"
WHISPER_MODEL = "base"
ASSEMBLYAI_API_KEY = "ada3df25b909471ca405dd86fc221940"
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
OUTPUT_DIR = "recorded_audio"
UPLOAD_DIR = "uploads"
RECORD_SECONDS = 30

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize MongoDB connection
try:
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    logger.info(f"Connected to MongoDB database: {DB_NAME}, collection: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    sys.exit(1)

# Initialize Whisper model
logger.info("Loading Whisper model...")
whisper_model = whisper.load_model(WHISPER_MODEL)
logger.info("Whisper model loaded!")

# Initialize KeyBERT model
logger.info("Loading KeyBERT model...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=sentence_model)
logger.info("KeyBERT model loaded!")

# Create FastAPI app
app = FastAPI(
    title="Audio Transcription API",
    description="API for transcribing audio from multiple sources, extracting keywords, and storing results",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="tts_api/static"), name="static")

# Pydantic models for request/response
class AudioSourceConfig(BaseModel):
    source_type: str = Field(..., description="Type of audio source ('microphone' or 'call')")
    source_name: str = Field(..., description="Name of the source")
    device_index: Optional[int] = Field(None, description="Audio device index")

class SpeakerMapping(BaseModel):
    speaker_id: str
    name: str

class TranscriptionResult(BaseModel):
    id: str
    source_type: str
    timestamp: str
    utterances: List[Dict[str, Any]]
    speaker_mapping: Dict[str, str]
    keywords: List[Dict[str, Any]]

class KeywordExtractionRequest(BaseModel):
    text: str
    top_n: int = 3
    method: str = "keybert"

class RecordingStatus(BaseModel):
    status: str
    session_id: Optional[str] = None
    message: str

class SimpleRecordingRequest(BaseModel):
    device_index: Optional[int] = None
    duration: Optional[int] = 30
    extract_keywords: Optional[bool] = True
    top_n: Optional[int] = 3

# Active recording sessions
active_recordings = {}

class RecordingSession:
    def __init__(self, device_index):
        self.device_index = device_index
        self.frames = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.thread = None

    def start(self):
        self.is_recording = True
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK
        )
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def stop(self):
        self.is_recording = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        return self.frames

    def _record(self):
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK)
                self.frames.append(data)
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                break

class StartRecordingRequest(BaseModel):
    device_index: Optional[int] = None
    extract_keywords: Optional[bool] = True
    top_n: Optional[int] = 3

# Helper class for audio processing
class AudioProcessor:
    @staticmethod
    def detect_speaker_names(utterances):
        """Detect speaker names from the conversation."""
        speaker_names = {}
        name_patterns = [
            r"(?:I am|I'm|this is|speaking is|name is) (\w+)",  # Matches "I am John", "I'm John", etc.
            r"(\w+) (?:speaking|here)",  # Matches "John speaking", "John here"
        ]
        
        for utterance in utterances:
            speaker = utterance["speaker"]
            text = utterance["text"]
            
            # Skip if we already found this speaker's name
            if speaker in speaker_names:
                continue
            
            # Try each pattern
            for pattern in name_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Take the first name found
                    name = matches[0].strip()
                    if len(name) > 1:  # Ensure name is at least 2 characters
                        speaker_names[speaker] = name
                        break
        
        return speaker_names

    @staticmethod
    def process_with_assemblyai(audio_file):
        """Process audio file with AssemblyAI for speaker diarization."""
        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json"
        }
        
        # Upload file
        logger.info(f"Sending file to AssemblyAI...")
        upload_headers = {
            "authorization": ASSEMBLYAI_API_KEY,
        }
        
        try:
            with open(audio_file, 'rb') as f:
                upload_response = requests.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=upload_headers,
                    data=f
                )
            
            if upload_response.status_code != 200:
                raise Exception(f"Error uploading file: {upload_response.text}")
            
            upload_url = upload_response.json()["upload_url"]
            logger.info(f"File uploaded successfully")
            
            # Start transcription with speaker diarization
            transcript_response = requests.post(
                "https://api.assemblyai.com/v2/transcript",
                headers=headers,
                json={
                    "audio_url": upload_url,
                    "speaker_labels": True,
                    "speakers_expected": 2
                }
            )
            
            if transcript_response.status_code != 200:
                raise Exception(f"Error starting transcription: {transcript_response.text}")
            
            transcript_id = transcript_response.json()["id"]
            logger.info(f"Transcription started with ID: {transcript_id}")
            
            # Wait for completion
            while True:
                polling_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers
                )
                
                polling_data = polling_response.json()
                status = polling_data["status"]
                
                if status == "completed":
                    logger.info(f"Transcription completed successfully")
                    break
                elif status == "error":
                    raise Exception(f"Transcription error: {polling_data}")
                
                time.sleep(3)
            
            # Extract utterances from the completed transcript
            utterances = []
            for utterance in polling_data.get("utterances", []):
                utterances.append({
                    "speaker": utterance["speaker"],
                    "text": utterance["text"],
                    "start": utterance["start"],
                    "end": utterance["end"]
                })
            
            if not utterances:
                logger.warning(f"No speaker-separated utterances found")
                utterances = [{
                    "speaker": "A",
                    "text": polling_data.get("text", ""),
                    "start": 0,
                    "end": 0
                }]
            
            # Detect and map speaker names
            speaker_names = AudioProcessor.detect_speaker_names(utterances)
            logger.info(f"Detected speaker names: {speaker_names}")
            
            return utterances, speaker_names
            
        except Exception as e:
            logger.error(f"AssemblyAI API error: {str(e)}")
            raise
    
    @staticmethod
    def extract_keywords(text, top_n=3, method='keybert'):
        """Extract keywords using KeyBERT."""
        try:
            # KeyBERT approach
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                use_maxsum=True,
                diversity=0.7
            )
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    @staticmethod
    def transcribe_with_whisper(audio_file):
        """Transcribe audio using local Whisper model."""
        try:
            result = whisper_model.transcribe(audio_file)
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing with Whisper: {str(e)}")
            raise

# API Routes
@app.get("/")
async def root():
    return {"message": "Audio Transcription API is running"}

@app.get("/api/devices")
async def get_devices():
    """Get available audio input devices."""
    try:
        audio = pyaudio.PyAudio()
        devices = []
        
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                devices.append({
                    'id': i,
                    'name': device_info.get('name', f'Device {i}'),
                    'channels': device_info.get('maxInputChannels'),
                    'sample_rate': int(device_info.get('defaultSampleRate', 44100))
                })
        
        audio.terminate()
        return {"devices": devices}
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/record/start")
async def start_recording(request: StartRecordingRequest):
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Initialize recording session
        audio = pyaudio.PyAudio()
        device_index = request.device_index if request.device_index is not None else audio.get_default_input_device_info()['index']
        audio.terminate()
        
        session = RecordingSession(device_index)
        active_recordings[session_id] = session
        session.start()
        
        return {"session_id": session_id, "status": "recording"}
    except Exception as e:
        logger.error(f"Error starting recording: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/record/stop/{session_id}")
async def stop_recording(
    session_id: str,
    extract_keywords: bool = Query(True, description="Whether to extract keywords from the transcription"),
    top_n: int = Query(3, ge=1, le=10, description="Number of keywords to extract")
):
    """
    Stop recording and process the audio.
    
    Parameters:
    - session_id: The ID of the recording session to stop
    - extract_keywords: Whether to extract keywords from the transcription
    - top_n: Number of keywords to extract (1-10)
    """
    try:
        logger.info(f"Stopping recording for session {session_id}")
        
        if session_id not in active_recordings:
            logger.error(f"Session {session_id} not found in active recordings")
            raise HTTPException(
                status_code=404,
                detail="Recording session not found. The session may have expired or been stopped already."
            )
        
        session = active_recordings[session_id]
        logger.info("Getting recorded frames...")
        frames = session.stop()
        
        if not frames:
            logger.error("No audio frames recorded")
            raise HTTPException(
                status_code=400,
                detail="No audio was recorded. Please check your microphone and try again."
            )
        
        # Save the recorded audio to a file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(OUTPUT_DIR, f"{file_id}.wav")
        logger.info(f"Saving audio to {file_path}")
        
        try:
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(session.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
        except Exception as e:
            logger.error(f"Error saving audio file: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save audio file: {str(e)}"
            )
        
        # Process with AssemblyAI for speaker diarization
        logger.info("Processing with AssemblyAI...")
        try:
            utterances, speaker_names = AudioProcessor.process_with_assemblyai(file_path)
            if not utterances:
                logger.warning("No utterances returned from AssemblyAI")
                # Fallback to Whisper
                raise Exception("No utterances found")
        except Exception as e:
            logger.error(f"AssemblyAI processing error: {str(e)}")
            # Fallback to Whisper if AssemblyAI fails
            logger.info("Falling back to Whisper transcription...")
            try:
                text = AudioProcessor.transcribe_with_whisper(file_path)
                if text.strip():
                    utterances = [{
                        "speaker": "A",
                        "text": text,
                        "start": 0,
                        "end": 0,
                        "confidence": 1.0
                    }]
                    speaker_names = {}
                else:
                    utterances = []
                    speaker_names = {}
            except Exception as whisper_error:
                logger.error(f"Whisper transcription error: {str(whisper_error)}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to transcribe audio with both AssemblyAI and Whisper. Please try again."
                )
        
        # Extract keywords if requested
        keywords = []
        if extract_keywords and utterances:
            logger.info(f"Extracting {top_n} keywords...")
            try:
                full_text = " ".join([u["text"] for u in utterances])
                if full_text.strip():
                    keywords = AudioProcessor.extract_keywords(full_text, top_n=top_n)
            except Exception as e:
                logger.error(f"Error extracting keywords: {str(e)}")
                # Continue without keywords
        
        # Create result document with speaker names
        result = {
            "source_type": "microphone",
            "timestamp": datetime.now(),
            "utterances": utterances,
            "speaker_mapping": speaker_names,  # Use detected speaker names
            "keywords": keywords,
            "audio_file": file_path
        }
        
        # Save to MongoDB
        logger.info("Saving to MongoDB...")
        try:
            insert_result = collection.insert_one(result)
            result_id = str(insert_result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB error: {str(e)}")
            # Continue without MongoDB storage
            result_id = str(uuid.uuid4())
        
        # Clean up
        logger.info("Cleaning up recording session...")
        del active_recordings[session_id]
        
        response_data = {
            "id": result_id,
            "source_type": "microphone",
            "timestamp": result["timestamp"].isoformat(),
            "utterances": utterances,
            "speaker_mapping": speaker_names,  # Include speaker names in response
            "keywords": keywords,
            "audio_file": file_path
        }
        
        if not utterances:
            logger.warning("No speech detected in the recording")
            return JSONResponse(
                status_code=200,
                content={
                    **response_data,
                    "message": "No speech was detected in the recording. Please try again with clearer audio."
                }
            )
        
        logger.info("Recording stopped and processed successfully")
        return response_data
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error stopping recording: {str(e)}")
        # Make sure to clean up the session even if there's an error
        if session_id in active_recordings:
            try:
                active_recordings[session_id].stop()
                del active_recordings[session_id]
            except:
                pass
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/transcribe/file", response_model=TranscriptionResult)
async def transcribe_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_type: str = Form("unknown"),
    extract_keywords: bool = Form(True),
    top_n: int = Form(3)
):
    """Transcribe an uploaded audio file using AssemblyAI."""
    try:
        # Save the uploaded file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{secure_filename(file.filename)}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process with AssemblyAI
        utterances = AudioProcessor.process_with_assemblyai(file_path)
        
        # Extract keywords if requested
        if extract_keywords:
            full_text = " ".join([u["text"] for u in utterances])
            keywords = AudioProcessor.extract_keywords(full_text, top_n=top_n)
        else:
            keywords = []
        
        # Create result document
        result = {
            "source_type": source_type,
            "timestamp": datetime.now(),
            "utterances": utterances,
            "speaker_mapping": {},  # Default mapping
            "keywords": keywords,
            "audio_file": file_path
        }
        
        # Save to MongoDB
        insert_result = collection.insert_one(result)
        result_id = str(insert_result.inserted_id)
        
        # Schedule cleanup of the temp file
        background_tasks.add_task(lambda: os.unlink(file_path))
        
        return {
            "id": result_id,
            "source_type": source_type,
            "timestamp": result["timestamp"].isoformat(),
            "utterances": utterances,
            "speaker_mapping": {},
            "keywords": keywords
        }
        
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/whisper", response_model=Dict[str, Any])
async def transcribe_with_whisper_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Transcribe audio using local Whisper model."""
    try:
        # Save the uploaded file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{secure_filename(file.filename)}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process with Whisper
        transcription = AudioProcessor.transcribe_with_whisper(file_path)
        
        # Schedule cleanup of the temp file
        background_tasks.add_task(lambda: os.unlink(file_path))
        
        return {
            "text": transcription,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing with Whisper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-keywords", response_model=Dict[str, Any])
async def extract_keywords_endpoint(request: KeywordExtractionRequest):
    """Extract keywords from text."""
    try:
        keywords = AudioProcessor.extract_keywords(
            request.text, 
            top_n=request.top_n, 
            method=request.method
        )
        
        return {
            "text": request.text,
            "keywords": keywords
        }
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recordings", response_model=List[Dict[str, Any]])
async def list_recordings(
    limit: int = Query(10, gt=0, le=100),
    skip: int = Query(0, ge=0),
    sort_by: str = Query("timestamp", enum=["timestamp", "source_type"]),
    sort_order: int = Query(-1, enum=[1, -1])
):
    """List recordings from MongoDB."""
    try:
        cursor = collection.find().sort(sort_by, sort_order).skip(skip).limit(limit)
        results = loads(dumps(list(cursor)))
        
        # Convert ObjectId to string
        for result in results:
            result["id"] = str(result["_id"])
            del result["_id"]
        
        return results
        
    except Exception as e:
        logger.error(f"Error listing recordings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recording/{recording_id}", response_model=Dict[str, Any])
async def get_recording(recording_id: str):
    """Get a specific recording by ID."""
    try:
        result = collection.find_one({"_id": ObjectId(recording_id)})
        if not result:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        result = loads(dumps(result))
        result["id"] = str(result["_id"])
        del result["_id"]
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting recording: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/recording/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recording(recording_id: str):
    """Delete a recording by ID."""
    try:
        result = collection.delete_one({"_id": ObjectId(recording_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        return None
        
    except Exception as e:
        logger.error(f"Error deleting recording: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/recording/{recording_id}/speaker-mapping", response_model=Dict[str, Any])
async def update_speaker_mapping(recording_id: str, mappings: List[SpeakerMapping]):
    """Update speaker mappings for a recording."""
    try:
        mapping_dict = {item.speaker_id: item.name for item in mappings}
        
        result = collection.update_one(
            {"_id": ObjectId(recording_id)},
            {"$set": {"speaker_mapping": mapping_dict}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        updated = collection.find_one({"_id": ObjectId(recording_id)})
        updated = loads(dumps(updated))
        updated["id"] = str(updated["_id"])
        del updated["_id"]
        
        return updated
        
    except Exception as e:
        logger.error(f"Error updating speaker mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
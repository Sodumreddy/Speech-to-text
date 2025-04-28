from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Query, Depends, status, Request, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, HTMLResponse
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
# Uncomment Twilio imports
from twilio.twiml.voice_response import VoiceResponse, Gather
from dotenv import load_dotenv
import json
import io
from vector_store import get_vector_store
from rag_engine import RAGEngine
from llm_rag import get_llm_rag_engine
import asyncio
from starlette.websockets import WebSocketState
import traceback

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants from environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "Speech-to-text")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "voice")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
CHUNK = int(os.getenv("CHUNK", "1024"))
FORMAT = int(os.getenv("FORMAT", "16"))
CHANNELS = int(os.getenv("CHANNELS", "2"))
RATE = int(os.getenv("RATE", "16000"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "recorded_audio")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "30"))

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

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
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

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
        """Detect speaker names from the conversation with backtracking support."""
        speaker_names = {}
        name_patterns = [
            r"(?:I am|I'm|this is|speaking is|name is) (\w+)",  # Matches "I am John", "I'm John", etc.
            r"(\w+) (?:speaking|here)",  # Matches "John speaking", "John here"
            r"(?:my name is|call me) (\w+)",  # Matches "my name is John", "call me John"
            r"(\w+) (?:is my name|here to help)",  # Matches "John is my name", "John here to help"
            r"(\w+):",  # Matches "John:" in dialogue
            r"(\w+)(?:\s+said|\s+asked|\s+replied|\s+responded)",  # Matches "John said", "John asked", etc.
            r"(\w+)(?:\s+needs|\s+wants|\s+requires)",  # Matches "John needs", "John wants", etc.
        ]
        
        # First pass: collect all potential names
        for utterance in utterances:
            speaker = utterance["speaker"]
            text = utterance["text"].lower()
            
            # Try each pattern
            for pattern in name_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Take the first name found
                    name = matches[0].strip()
                    if len(name) > 1:  # Ensure name is at least 2 characters
                        speaker_names[speaker] = name
                        break
        
        # Second pass: look for direct references to speakers
        for utterance in utterances:
            speaker = utterance["speaker"]
            text = utterance["text"].lower()
            
            # Skip if we already have a name for this speaker
            if speaker in speaker_names:
                continue
                
            # Look for references to other speakers
            for other_speaker, name in speaker_names.items():
                if other_speaker != speaker and name.lower() in text:
                    # If this utterance mentions another speaker's name, it might be introducing the current speaker
                    for pattern in name_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            new_name = matches[0].strip()
                            if len(new_name) > 1 and new_name.lower() != name.lower():
                                speaker_names[speaker] = new_name
                                break
        
        # Third pass: look for contextual clues about who is speaking
        for i, utterance in enumerate(utterances):
            speaker = utterance["speaker"]
            text = utterance["text"].lower()
            
            # Skip if we already have a name for this speaker
            if speaker in speaker_names:
                continue
                
            # Look for job search related content
            if "job" in text and ("search" in text or "help" in text or "need" in text):
                # Check if this speaker mentioned needing help with job search
                if "need" in text and "help" in text and "job" in text:
                    # Look for a name in this utterance
                    for pattern in name_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            name = matches[0].strip()
                            if len(name) > 1:
                                speaker_names[speaker] = name
                                break
            
            # Look for medical-related content
            if "medical" in text or "doctor" in text or "patient" in text:
                # Check if this speaker mentioned helping with medical
                if "help" in text and ("medical" in text or "health" in text):
                    # Look for a name in this utterance
                    for pattern in name_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            name = matches[0].strip()
                            if len(name) > 1:
                                speaker_names[speaker] = name
                                break
        
        # Fourth pass: look for name mentions in the same utterance as "I am" or similar
        for utterance in utterances:
            speaker = utterance["speaker"]
            text = utterance["text"].lower()
            
            # Skip if we already have a name for this speaker
            if speaker in speaker_names:
                continue
            
            # Check if this utterance contains both a name and "I am" or similar
            for pattern in name_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    name = matches[0].strip()
                    if len(name) > 1:
                        # Check if this utterance also contains "I am" or similar
                        if re.search(r"(?:I am|I'm|this is|speaking is|name is|my name is|call me)", text, re.IGNORECASE):
                            speaker_names[speaker] = name
                            break
        
        # Fifth pass: look for name mentions in adjacent utterances
        for i, utterance in enumerate(utterances):
            speaker = utterance["speaker"]
            text = utterance["text"].lower()
            
            # Skip if we already have a name for this speaker
            if speaker in speaker_names:
                continue
            
            # Check if the previous utterance contains a name
            if i > 0:
                prev_utterance = utterances[i-1]
                prev_text = prev_utterance["text"].lower()
                
                for pattern in name_patterns:
                    matches = re.findall(pattern, prev_text, re.IGNORECASE)
                    if matches:
                        name = matches[0].strip()
                        if len(name) > 1:
                            # Check if this utterance contains "I am" or similar
                            if re.search(r"(?:I am|I'm|this is|speaking is|name is|my name is|call me)", text, re.IGNORECASE):
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
        
        try:
            # Upload the file
            with open(audio_file, 'rb') as f:
                upload_response = requests.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers={"authorization": ASSEMBLYAI_API_KEY},
                    data=f
                )
            
            if upload_response.status_code != 200:
                raise Exception(f"Error uploading file: {upload_response.text}")
            
            upload_url = upload_response.json()["upload_url"]
            logger.info(f"File uploaded successfully to AssemblyAI")
            
            # Start transcription with speaker diarization
            transcript_response = requests.post(
                "https://api.assemblyai.com/v2/transcript",
                headers=headers,
                json={
                    "audio_url": upload_url,
                    "speaker_labels": True,
                    "speakers_expected": 2  # Set expected number of speakers
                }
            )
            
            if transcript_response.status_code != 200:
                raise Exception(f"Error starting transcription: {transcript_response.text}")
            
            transcript_id = transcript_response.json()["id"]
            logger.info(f"AssemblyAI transcription started with ID: {transcript_id}")
            
            # Wait for completion
            polling_attempts = 0
            max_polling_attempts = 30  # Limit polling attempts
            
            while polling_attempts < max_polling_attempts:
                polling_attempts += 1
                
                polling_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers
                )
                
                polling_data = polling_response.json()
                status = polling_data["status"]
                
                if status == "completed":
                    logger.info(f"AssemblyAI transcription completed successfully after {polling_attempts} polling attempts")
                    break
                elif status == "error":
                    raise Exception(f"AssemblyAI transcription error: {polling_data}")
                
                # Wait before polling again, increasing the wait time slightly each time
                wait_time = min(3 + (polling_attempts * 0.5), 10)  # Start at 3s, gradually increase, cap at 10s
                logger.info(f"Waiting for AssemblyAI transcription to complete. Status: {status}, attempt {polling_attempts}")
                time.sleep(wait_time)
            
            if polling_attempts >= max_polling_attempts:
                raise Exception(f"AssemblyAI transcription timed out after {max_polling_attempts} polling attempts")
            
            # Extract utterances from the completed transcript
            utterances = []
            for utterance in polling_data.get("utterances", []):
                utterance_data = {
                    "speaker": utterance["speaker"],
                    "text": utterance["text"],
                    "start": utterance["start"],
                    "end": utterance["end"]
                }
                utterances.append(utterance_data)
                logger.info(f"Utterance: {utterance_data['speaker']} - '{utterance_data['text']}'")
            
            if not utterances:
                logger.warning(f"No speaker-separated utterances found in AssemblyAI response")
                utterances = [{
                    "speaker": "A",
                    "text": polling_data.get("text", ""),
                    "start": 0,
                    "end": 0
                }]
            
            # Detect and map speaker names
            speaker_names = AudioProcessor.detect_speaker_names(utterances)
            logger.info(f"Detected speaker names from utterances: {speaker_names}")
            
            # Return results
            return utterances, speaker_names
            
        except Exception as e:
            logger.error(f"AssemblyAI API error: {str(e)}")
            raise
    
    @staticmethod
    def extract_keywords(text, top_n=None, method='keybert'):
        """Extract keywords using KeyBERT with dynamic settings based on content."""
        try:
            # Calculate optimal number of keywords based on text length
            if top_n is None:
                # Count words in text
                word_count = len(text.split())
                # Base number of keywords on text length
                if word_count < 50:
                    top_n = 3
                elif word_count < 100:
                    top_n = 5
                elif word_count < 200:
                    top_n = 8
                elif word_count < 500:
                    top_n = 12
                else:
                    top_n = 15
            
            # First pass: extract main keywords with high confidence
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),  # Allow 1-3 word phrases
                stop_words='english',
                top_n=top_n,
                use_maxsum=True,
                diversity=0.5,  # Lower diversity to get more relevant keywords
                nr_candidates=20  # Consider more candidates
            )
            
            # Filter keywords by confidence score
            min_confidence = 0.3  # Minimum confidence threshold
            filtered_keywords = [kw for kw in keywords if kw[1] >= min_confidence]
            
            # If we have too few keywords, try with different settings
            if len(filtered_keywords) < top_n // 2:  # If we have less than half the target
                additional_keywords = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=top_n - len(filtered_keywords),
                    use_mmr=True,  # Use MMR for additional keywords
                    diversity=0.7
                )
                # Add only keywords that meet the confidence threshold
                filtered_keywords.extend([kw for kw in additional_keywords if kw[1] >= min_confidence])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for kw in filtered_keywords:
                if kw[0] not in seen:
                    seen.add(kw[0])
                    unique_keywords.append(kw)
            
            # Sort by confidence score
            unique_keywords.sort(key=lambda x: x[1], reverse=True)
            
            return unique_keywords
            
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
    extract_keywords: bool = Query(True, description="Whether to extract keywords from the transcription")
):
    """
    Stop recording and process the audio.
    
    Parameters:
    - session_id: The ID of the recording session to stop
    - extract_keywords: Whether to extract keywords from the transcription
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
            logger.info(f"Extracting keywords...")
            try:
                full_text = " ".join([u["text"] for u in utterances])
                if full_text.strip():
                    keywords = AudioProcessor.extract_keywords(full_text)
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
        
        # Add to vector database
        try:
            vector_store = get_vector_store()
            vector_store.add_transcription(result)
        except Exception as e:
            logger.error(f"Error indexing transcription in vector DB: {str(e)}")
            # Continue even if indexing fails
        
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

@app.get("/api/recordings/{recording_id}", response_model=Dict[str, Any])
async def get_recording(recording_id: str):
    """Get a specific recording by ID."""
    try:
        from bson.objectid import ObjectId
        
        # Try to find by ObjectId first
        try:
            obj_id = ObjectId(recording_id)
            recording = collection.find_one({"_id": obj_id})
        except:
            # If not a valid ObjectId, try to find by call_id
            recording = collection.find_one({"call_id": recording_id})
            
        if not recording:
            # Try to find by source_name containing the recording_id
            recording = collection.find_one({"source_name": {"$regex": recording_id, "$options": "i"}})
            
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")
            
        # Convert ObjectId to string for JSON serialization
        if "_id" in recording:
            recording["_id"] = str(recording["_id"])
        
        logger.info(f"Retrieved recording: {recording.get('source_type')} - {recording.get('source_name')}")
            
        return recording
    except HTTPException:
        raise
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

# === Voice Endpoint for Twilio Call Handling ===
@app.post("/voice")
async def voice_endpoint(request: Request):
    """Handle incoming Twilio calls and start recording automatically."""
    try:
        response = VoiceResponse()
        
        # Get the ngrok URL from the request
        host = request.headers.get("host", "")
        scheme = request.url.scheme
        base_url = f"{scheme}://{host}"
        
        # Configure Twilio Stream first
        ws_protocol = 'wss' if scheme == 'https' else 'ws'
        ws_url = f'{ws_protocol}://{host}/ws/transcription'
        
        # Welcome message
        response.say(
            "Welcome to the transcription service. Your call will be recorded and transcribed.",
            voice="alice"
        )
        
        # Configure media stream
        connect = response.connect()
        stream = connect.stream(
            name='stream_track',
            url=ws_url,
            track='inbound_track',
            parameters={
                'format': 'audio/x-mulaw',
                'sampleRate': '8000'
            }
        )
        
        # Start recording
        response.record(
            action=f'{base_url}/recording-complete',
            playBeep=False,
            trim='trim-silence',
            recordingStatusCallback=f'{base_url}/recording-status',
            recordingStatusCallbackMethod='POST',
            recordingChannels='mono',
            recordingSampleRate=8000
        )
        
        logger.info(f"Twilio call configured with WebSocket URL: {ws_url}")
        return Response(content=str(response), media_type="application/xml")
        
    except Exception as e:
        logger.error(f"Error in voice endpoint: {str(e)}")
        error_response = VoiceResponse()
        error_response.say("We're sorry, but there was an error processing your call.")
        return Response(content=str(error_response), media_type="application/xml")

@app.post("/recording-status")
async def recording_status(request: Request):
    """Handle recording status updates."""
    try:
        form_data = await request.form()
        recording_status = form_data.get("RecordingStatus")
        recording_sid = form_data.get("RecordingSid")
        call_sid = form_data.get("CallSid")
        
        logger.info(f"Recording status update - Status: {recording_status}, SID: {recording_sid}, Call SID: {call_sid}")
        
        # Store status update in MongoDB
        status_doc = {
            "recording_sid": recording_sid,
            "call_sid": call_sid,
            "status": recording_status,
            "timestamp": datetime.utcnow(),
            "source_type": "twilio_call"
        }
        collection.insert_one(status_doc)
        
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logger.error(f"Error in recording status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/recording-complete")
async def recording_complete(request: Request):
    """Handle recording completion."""
    try:
        form_data = await request.form()
        recording_url = form_data.get("RecordingUrl")
        recording_sid = form_data.get("RecordingSid")
        call_sid = form_data.get("CallSid")
        duration = form_data.get("RecordingDuration")
        
        logger.info(f"Recording completed - SID: {recording_sid}, Duration: {duration}s, URL: {recording_url}")
        
        # Store recording details in MongoDB
        recording_doc = {
            "recording_sid": recording_sid,
            "call_sid": call_sid,
            "recording_url": recording_url,
            "duration": duration,
            "timestamp": datetime.utcnow(),
            "source_type": "twilio_call",
            "status": "completed"
        }
        collection.insert_one(recording_doc)
        
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logger.error(f"Error in recording complete: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# === WebSocket Endpoint for Streaming Audio Data ===
@app.websocket("/ws/transcription")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription from various audio sources."""
    await websocket.accept()
    logger.info("WebSocket connection established for transcription")
    
    # Additional state variable to track connection status
    connection_active = True
    
    try:
        # Wait for initial message
        data = await websocket.receive_text()
        msg = json.loads(data)
        
        if msg.get("type") == "start":
            logger.info("Received start message from client")
            # Get configuration from message
            config = msg.get("config", {})
            use_llm = config.get("use_llm", True) 
            extract_keywords = config.get("extract_keywords", True)
            
            # Get source information from the config
            source_type = config.get("source_type", "twilio_call")
            source_name = config.get("source_name", source_type)
            
            logger.info(f"Audio source: {source_type}, Name: {source_name}")
            
            # Send confirmation
            await websocket.send_json({"status": "started", "source_type": source_type})
            
            # Set up initial state for this client
            current_transcription = []
            current_keywords = []
            audio_buffer = []
            call_id = str(uuid.uuid4())  # Generate call_id immediately
            last_insight_time = time.time()
            insight_interval = 30  # Generate insights every 30 seconds
            current_full_text = ""
            
            # Create a dummy document to represent the session even before audio
            try:
                dummy_doc = {
                    "call_id": call_id,
                    "timestamp": datetime.utcnow(),
                    "utterances": [],
                    "keywords": [],
                    "source_type": source_type,
                    "source_name": source_name,
                    "status": "started"
                }
                collection.insert_one(dummy_doc)
                logger.info(f"Started new {source_type} session: {call_id}")
                
                # Send the call_id to client
                await websocket.send_json({
                    "type": "session_started",
                    "call_id": call_id,
                    "source_type": source_type
                })
            except Exception as e:
                logger.error(f"Error creating session document: {str(e)}")
            
            # Keep connection alive and simulate some responses for browser clients
            if source_type != "twilio_call":
                # For Zoom/Teams/etc. sources (browser-based), provide a simulated experience
                # until we implement actual browser microphone capture
                await simulate_transcription_session(websocket, call_id, source_type, source_name, use_llm)
            else:  
                # For Twilio calls, process the actual streaming audio coming in
                while connection_active:
                    try:
                        # Non-blocking check for messages with shorter timeout
                        message = await asyncio.wait_for(websocket.receive(), timeout=0.5)
                        
                        if "text" in message:
                            # Text message (e.g. control commands)
                            data = json.loads(message["text"])
                            if data.get("type") == "stop":
                                logger.info(f"Received stop message from client for {source_type}")
                                await websocket.send_json({"status": "stopped"})
                                
                                # Update the session status
                                try:
                                    collection.update_one(
                                        {"call_id": call_id},
                                        {"$set": {"status": "completed"}}
                                    )
                                except Exception as e:
                                    logger.error(f"Error updating session status: {str(e)}")
                                
                                connection_active = False
                                break
                        
                        elif "bytes" in message:
                            # Binary message (audio data)
                            audio_chunk = message["bytes"]
                            audio_buffer.append(audio_chunk)
                            
                            # When buffer reaches certain size, process it
                            if len(audio_buffer) >= 5:  # Process every ~0.5 seconds for better responsiveness
                                # Process audio buffer in the background with source info
                                new_utterances, new_keywords = await process_audio_buffer(
                                    audio_buffer, 
                                    call_id, 
                                    source_type=source_type,
                                    source_name=source_name
                                )
                                audio_buffer = []  # Clear buffer after processing
                                
                                if new_utterances:
                                    # Add current utterances to transcription
                                    current_transcription.extend(new_utterances)
                                    
                                    # Update keywords
                                    if new_keywords:
                                        current_keywords = new_keywords
                                    
                                    # Update the full text for insights
                                    current_full_text = " ".join([u["text"] for u in current_transcription])
                                    
                                    # Send updated transcription to client
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "utterances": new_utterances,
                                        "keywords": current_keywords,
                                        "call_id": call_id,
                                        "source_type": source_type
                                    })
                                    
                                    # Check if it's time to generate insights and LLM is enabled
                                    current_time = time.time()
                                    if use_llm and current_time - last_insight_time >= insight_interval:
                                        # Only generate insights if we have enough text
                                        if len(current_full_text) > 50:
                                            # Run in a separate task to avoid blocking
                                            asyncio.create_task(generate_and_send_insights(
                                                websocket, 
                                                current_full_text, 
                                                current_keywords, 
                                                call_id,
                                                source_type
                                            ))
                                            
                                            # Update last insight time
                                            last_insight_time = current_time
                        
                    except asyncio.TimeoutError:
                        # No new message received, check if connection is still alive
                        try:
                            # Send a ping message to keep connection alive
                            await websocket.send_json({"type": "ping"})
                        except Exception as e:
                            logger.error(f"Error sending ping: {str(e)}")
                            connection_active = False
                            break
                    except WebSocketDisconnect:
                        logger.info(f"WebSocket disconnected for {source_type}")
                        connection_active = False
                        break
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        if "disconnect" in str(e).lower():
                            connection_active = False
                            break
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0.1)
            
            # Mark connection as completed when loop exits
            logger.info(f"WebSocket session for {source_type} completed")
        else:
            logger.warning(f"Unexpected initial message: {msg}")
            await websocket.send_json({"type": "error", "message": "Expected 'start' message"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    
    logger.info("WebSocket endpoint function completed")

async def simulate_transcription_session(websocket, call_id, source_type, source_name, use_llm=True):
    """
    Handle browser-based audio capture for sources like Zoom and Teams.
    Process real audio data from the browser using MediaRecorder format.
    """
    # State variables
    connection_active = True
    accumulated_audio = []
    accumulated_size = 0
    min_accumulation_seconds = 20  # Accumulate at least 20 seconds of audio
    
    try:
        # Send a welcome message
        welcome_utterance = [{
            "speaker": "System",
            "text": f"Transcription session started for {source_name}. Start speaking to see transcription... (First transcription may take 20-30 seconds)",
            "start": time.time(),
            "end": time.time()
        }]
        
        await websocket.send_json({
            "type": "transcription",
            "utterances": welcome_utterance,
            "keywords": [],
            "call_id": call_id,
            "source_type": source_type
        })
        
        # Initial state
        current_transcription = []
        current_keywords = []
        current_speaker_mapping = {}
        last_insight_time = time.time()
        insight_interval = 30
        current_full_text = ""
        
        # Start time for accumulation
        accumulation_start_time = time.time()
        
        # Keep the connection open and receive audio data
        while connection_active:
            try:
                # Check for messages
                message = await asyncio.wait_for(websocket.receive(), timeout=2)
                
                if "text" in message:
                    # Text message (control commands)
                    data = json.loads(message["text"])
                    if data.get("type") == "stop":
                        logger.info(f"Received stop message from client for {source_type}")
                        try:
                            await websocket.send_json({"status": "stopped"})
                        except Exception:
                            pass
                        
                        # Update session status
                        try:
                            collection.update_one(
                                {"call_id": call_id},
                                {"$set": {"status": "completed"}}
                            )
                        except Exception as e:
                            logger.error(f"Error updating session status: {str(e)}")
                        
                        connection_active = False
                        break
                
                elif "bytes" in message:
                    # Binary message (audio data)
                    audio_chunk = message["bytes"]
                    accumulated_audio.append(audio_chunk)
                    accumulated_size += len(audio_chunk)
                    
                    # Check if we've accumulated enough audio data
                    current_time = time.time()
                    elapsed_time = current_time - accumulation_start_time
                    
                    # Process audio when either:
                    # 1. We've recorded for at least min_accumulation_seconds (20 seconds) AND have significant data
                    # 2. We've accumulated a huge amount of data regardless of time
                    # 3. It's been over 30 seconds since last processing
                    should_process = (
                        (elapsed_time >= min_accumulation_seconds and accumulated_size > 200000) or  # 20 sec + 200KB+
                        (accumulated_size > 500000) or  # Huge data (500KB+)
                        (elapsed_time > 30)  # Over 30 seconds since last processing
                    )
                    
                    if should_process and accumulated_audio:
                        logger.info(f"Processing accumulated audio: {len(accumulated_audio)} chunks, "
                                  f"{accumulated_size/1024:.1f}KB, after {elapsed_time:.1f} seconds")
                        
                        # Process the accumulated audio
                        new_utterances, new_keywords = await process_browser_audio(
                            accumulated_audio, 
                            call_id, 
                            source_type=source_type,
                            source_name=source_name
                        )
                        
                        # Reset accumulation
                        accumulated_audio = []
                        accumulated_size = 0
                        accumulation_start_time = current_time
                        
                        if new_utterances:
                            # Add to current transcription
                            current_transcription.extend(new_utterances)
                            
                            # Update keywords
                            if new_keywords:
                                current_keywords = new_keywords
                            
                            # Get latest speaker mapping
                            try:
                                latest_doc = collection.find_one(
                                    {"call_id": call_id},
                                    sort=[("timestamp", -1)]
                                )
                                if latest_doc and "speaker_mapping" in latest_doc:
                                    current_speaker_mapping = latest_doc["speaker_mapping"]
                            except Exception as e:
                                logger.error(f"Error retrieving speaker mapping: {str(e)}")
                            
                            # Update full text for insights
                            current_full_text = " ".join([u["text"] for u in current_transcription])
                            
                            # Send updated transcription
                            await websocket.send_json({
                                "type": "transcription",
                                "utterances": new_utterances,
                                "keywords": current_keywords,
                                "speaker_mapping": current_speaker_mapping,
                                "call_id": call_id,
                                "source_type": source_type
                            })
                            
                            # Check if it's time to generate insights
                            if use_llm and current_time - last_insight_time >= insight_interval:
                                if len(current_full_text) > 50:
                                    # Generate insights in separate task
                                    asyncio.create_task(generate_and_send_insights(
                                        websocket, 
                                        current_full_text, 
                                        current_keywords, 
                                        call_id,
                                        source_type
                                    ))
                                    last_insight_time = current_time
                
            except asyncio.TimeoutError:
                # Keep connection alive with ping
                try:
                    await websocket.send_json({"type": "ping"})
                    
                    # Check if we should process accumulated data after timeout
                    current_time = time.time()
                    elapsed_time = current_time - accumulation_start_time
                    
                    # Process if we have enough data and it's been a while
                    if elapsed_time >= min_accumulation_seconds and accumulated_size > 100000:
                        logger.info(f"Processing audio after timeout: {accumulated_size/1024:.1f}KB, "
                                  f"{elapsed_time:.1f} seconds elapsed")
                        
                        new_utterances, new_keywords = await process_browser_audio(
                            accumulated_audio, 
                            call_id, 
                            source_type=source_type,
                            source_name=source_name
                        )
                        
                        # Reset accumulation
                        accumulated_audio = []
                        accumulated_size = 0
                        accumulation_start_time = current_time
                        
                        if new_utterances:
                            # Update speaker mapping
                            try:
                                latest_doc = collection.find_one(
                                    {"call_id": call_id},
                                    sort=[("timestamp", -1)]
                                )
                                if latest_doc and "speaker_mapping" in latest_doc:
                                    current_speaker_mapping = latest_doc["speaker_mapping"]
                            except Exception as e:
                                logger.error(f"Error retrieving speaker mapping: {str(e)}")
                            
                            # Send updated transcription
                            await websocket.send_json({
                                "type": "transcription",
                                "utterances": new_utterances,
                                "keywords": new_keywords,
                                "speaker_mapping": current_speaker_mapping,
                                "call_id": call_id,
                                "source_type": source_type
                            })
                except Exception as e:
                    logger.error(f"Error during ping: {str(e)}")
                    connection_active = False
                    break
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for {source_type}")
                connection_active = False
                break
            except Exception as e:
                logger.error(f"Error in audio session: {str(e)}")
                if "disconnect" in str(e).lower():
                    connection_active = False
                    break
            
            # Allow other tasks to run
            await asyncio.sleep(0.1)
        
        logger.info(f"Audio session ended for {source_type} {call_id}")
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {source_type}")
    except Exception as e:
        logger.error(f"Error in audio session: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    
    logger.info(f"Exiting audio function for {source_type} {call_id}")

async def generate_and_send_insights(websocket, transcription_text, keywords, call_id, source_type="twilio_call"):
    """Generate and send LLM insights to the client."""
    try:
        # Get LLM RAG engine
        llm_rag = get_llm_rag_engine()
        
        # Extract keywords for the query
        keyword_terms = [k[0] for k in keywords] if keywords else []
        
        # Create a task for the LLM processing so it doesn't block
        async def process_insights():
            try:
                # Log start of processing
                logger.info(f"Starting insights generation for {source_type} {call_id}")
                
                # Generate insights
                insights = llm_rag.answer_from_live_transcription(
                    transcription_text=transcription_text,
                    keywords=keyword_terms
                )
                
                # Process and send back the results only if the websocket is still open
                if websocket.client_state == WebSocketState.CONNECTED:
                    # Send insights to client if successful
                    if not insights.get("error", False):
                        await websocket.send_json({
                            "type": "insights",
                            "insights": insights["response"],
                            "call_id": call_id,
                            "source_type": source_type
                        })
                        logger.info(f"Sent LLM insights for {source_type} {call_id}")
                    else:
                        logger.error(f"Error generating insights: {insights.get('response')}")
                else:
                    logger.warning("WebSocket closed before insights could be sent")
            except Exception as e:
                logger.error(f"Error in async insights processing: {str(e)}")
        
        # Schedule the task without waiting for it to complete
        asyncio.create_task(process_insights())
        
        # Send a notification that insights are being generated
        try:
            await websocket.send_json({
                "type": "insights_status",
                "status": "generating",
                "message": "Generating insights from conversation..."
            })
        except Exception as e:
            logger.warning(f"Error sending insights status: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error setting up insights generation: {str(e)}")
        # Don't propagate the exception - this shouldn't interrupt the main flow

# Modify process_audio_buffer function to use AssemblyAI for speaker diarization
async def process_audio_buffer(audio_buffer, call_id, source_type="twilio_call", source_name=None):
    """Process audio buffer and return transcription with keywords."""
    try:
        # Convert buffer to WAV format
        wav_bytes = convert_audio_to_wav(audio_buffer)
        
        # Save to temporary file
        temp_file = os.path.join(OUTPUT_DIR, f"temp_{call_id}.wav")
        with open(temp_file, "wb") as f:
            f.write(wav_bytes)
        
        # Try using AssemblyAI for speaker diarization first
        try:
            # Process with AssemblyAI for speaker diarization
            logger.info(f"Processing audio with AssemblyAI for speaker diarization...")
            utterances, speaker_names = AudioProcessor.process_with_assemblyai(temp_file)
            
            if utterances:
                # Extract keywords from the full text
                full_text = " ".join([u["text"] for u in utterances])
                keywords = []
                if len(full_text.split()) > 3:
                    keywords = AudioProcessor.extract_keywords(full_text)
                    logger.info(f"Extracted keywords: {keywords}")
                
                # Update MongoDB (store transcription segment)
                try:
                    doc = {
                        "call_id": call_id,
                        "timestamp": datetime.utcnow(),
                        "utterances": utterances,
                        "speaker_mapping": speaker_names,
                        "keywords": keywords,
                        "source_type": source_type,
                        "source_name": source_name or source_type,
                        "audio_file": temp_file
                    }
                    collection.insert_one(doc)
                    logger.info(f"Stored transcription segment with speaker diarization for {source_type} {call_id}")
                except Exception as e:
                    logger.error(f"Error storing transcription: {str(e)}")
                
                return utterances, keywords
            else:
                logger.warning("No utterances returned from AssemblyAI, falling back to Whisper")
        except Exception as e:
            logger.error(f"AssemblyAI processing error: {str(e)}")
            logger.info("Falling back to Whisper for transcription without diarization")
        
        # Fallback to Whisper if AssemblyAI fails or returns no utterances
        # Transcribe with Whisper
        result = whisper_model.transcribe(temp_file, language="en")
        
        if result and result["text"]:
            # Clean up the text
            text = result["text"].strip()
            
            # Create utterance
            utterance = {
                "speaker": "Speaker",
                "text": text,
                "start": time.time(),
                "end": time.time()
            }
            
            # Extract keywords
            keywords = AudioProcessor.extract_keywords(text)
            
            # Create final utterances list
            utterances = [utterance]
            
            # Update MongoDB (store transcription segment)
            try:
                doc = {
                    "call_id": call_id,
                    "timestamp": datetime.utcnow(),
                    "utterances": utterances,
                    "keywords": keywords,
                    "source_type": source_type,
                    "source_name": source_name or source_type,
                    "audio_file": temp_file
                }
                collection.insert_one(doc)
                logger.info(f"Stored transcription segment for {source_type} {call_id}")
            except Exception as e:
                logger.error(f"Error storing transcription: {str(e)}")
            
            return utterances, keywords
        
        return [], []
        
    except Exception as e:
        logger.error(f"Error processing audio buffer: {str(e)}")
        return [], []

def convert_audio_to_wav(audio_buffer):
    """Convert Twilio audio format (mulaw) to WAV format."""
    try:
        import wave
        import audioop
        
        # Combine all audio chunks
        audio_data = b''.join(audio_buffer)
        
        # Convert from mulaw to PCM
        pcm_data = audioop.ulaw2lin(audio_data, 2)  # 2 bytes per sample
        
        # Create WAV data in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(8000)  # 8kHz sample rate (Twilio's rate)
            wav_file.writeframes(pcm_data)
        
        return wav_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error converting audio format: {str(e)}")
        raise

@app.get("/api/recordings")
async def list_recordings(source: Optional[str] = None):
    """List all recordings with optional source filter."""
    try:
        query = {}
        if source:
            query["source_type"] = source
            
        recordings = list(collection.find(query).sort("timestamp", -1))
        
        # Convert ObjectId to string for JSON serialization
        for recording in recordings:
            recording["_id"] = str(recording["_id"])
            
        return {"recordings": recordings}
    except Exception as e:
        logger.error(f"Error listing recordings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recording/{recording_id}", response_model=Dict[str, Any])
async def get_recording_old(recording_id: str):
    """Get a specific recording by ID (DEPRECATED - Use /api/recordings/{recording_id} instead)."""
    logger.warning("Using deprecated endpoint /recording/{recording_id}. Use /api/recordings/{recording_id} instead.")
    # Forward to new endpoint
    return await get_recording(recording_id)

# Add a test route
@app.get("/test")
async def test():
    return {"message": "Application is working!"}

# Add configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get application configuration."""
    return {
        "twilio_number": TWILIO_PHONE_NUMBER,
        "websocket_enabled": True
    }

# === RAG Endpoints ===
@app.post("/api/rag/index")
async def index_transcription(transcription_id: str):
    """Index a transcription in the vector database."""
    try:
        # Get transcription from MongoDB
        transcription = collection.find_one({"_id": ObjectId(transcription_id)})
        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Add to vector database
        doc_id = vector_store.add_transcription(transcription)
        
        return {
            "status": "success",
            "message": f"Transcription {transcription_id} indexed successfully",
            "doc_id": doc_id
        }
    except Exception as e:
        logger.error(f"Error indexing transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/search", response_class=JSONResponse)
async def search_transcriptions(
    query: str = Query(..., description="Search query"),
    n_results: int = Query(5, description="Number of results to return"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    min_date: Optional[str] = Query(None, description="Filter by minimum date (ISO format)"),
    max_date: Optional[str] = Query(None, description="Filter by maximum date (ISO format)")
):
    """Search transcriptions using vector embeddings."""
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Perform search
        results = vector_store.search(
            query=query,
            n_results=n_results,
            source_type=source_type,
            min_date=min_date,
            max_date=max_date
        )
        
        # Return results
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching transcriptions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching transcriptions: {str(e)}")

@app.get("/api/rag/similar/{transcription_id}", response_class=JSONResponse)
async def get_similar_transcriptions(
    transcription_id: str,
    n_results: int = Query(5, description="Number of similar transcriptions to return")
):
    """Find transcriptions similar to a given one."""
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Find similar transcriptions
        similar = vector_store.get_similar_transcriptions(
            transcription_id=transcription_id,
            n_results=n_results
        )
        
        # Return results
        return {"similar_transcriptions": similar}
    except Exception as e:
        logger.error(f"Error finding similar transcriptions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding similar transcriptions: {str(e)}")

@app.get("/api/rag/topics", response_class=JSONResponse)
async def get_topics(max_topics: int = Query(20, description="Maximum number of topics to return")):
    """Get common topics across all transcriptions."""
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Extract topics from transcriptions
        topics = vector_store.extract_common_topics(max_topics=max_topics)
        
        return {
            "topics": topics,
            "total_transcriptions": vector_store.get_collection_stats()["count"]
        }
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting topics: {str(e)}")

@app.get("/api/rag/stats", response_class=JSONResponse)
async def get_rag_statistics():
    """Get statistics about the transcriptions in the vector store."""
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Get collection stats
        collection_stats = vector_store.get_collection_stats()
        
        # Get source distribution
        source_distribution = vector_store.get_source_distribution()
        
        # Get speaker statistics
        speaker_stats = vector_store.get_speaker_statistics()
        
        # Return combined statistics
        return {
            "collection": collection_stats,
            "sources": source_distribution,
            "speakers": speaker_stats
        }
    except Exception as e:
        logger.error(f"Error getting RAG statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting RAG statistics: {str(e)}")

@app.get("/api/rag/by-date", response_class=JSONResponse)
async def get_transcriptions_by_date(
    start_date: str = Query(..., description="Start date in ISO format (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date in ISO format (YYYY-MM-DD)"),
    limit: int = Query(100, description="Maximum number of transcriptions to return")
):
    """Get transcriptions within a date range."""
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Get transcriptions by date range
        transcriptions = vector_store.get_transcriptions_by_date_range(
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "count": len(transcriptions),
            "transcriptions": transcriptions
        }
    except Exception as e:
        logger.error(f"Error getting transcriptions by date: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting transcriptions by date: {str(e)}")

# Additional LLM RAG endpoints
@app.post("/api/rag/query", response_class=JSONResponse)
async def rag_query(
    query: str = Body(..., description="Query to answer"),
    context: Optional[str] = Body(None, description="Optional additional context"),
    n_results: int = Body(3, description="Number of results to fetch from vector store"),
    temperature: float = Body(0.7, description="LLM temperature parameter"),
    max_tokens: int = Body(1024, description="Maximum tokens to generate")
):
    """Query the LLM with RAG context."""
    try:
        # Get LLM RAG engine
        llm_rag = get_llm_rag_engine()
        
        # Query the LLM
        result = llm_rag.query(
            query=query,
            context=context,
            n_results=n_results,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Remove raw response to avoid huge payloads
        if "raw_response" in result:
            del result["raw_response"]
        
        return result
    except Exception as e:
        logger.error(f"Error querying LLM with RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying LLM with RAG: {str(e)}")

@app.post("/api/rag/insights", response_class=JSONResponse)
async def extract_insights(
    transcription_text: str = Body(..., description="Transcription text to analyze"),
    keywords: Optional[List[str]] = Body(None, description="Optional keywords to focus on")
):
    """Extract insights from transcription text."""
    try:
        # Get LLM RAG engine
        llm_rag = get_llm_rag_engine()
        
        # Extract insights
        result = llm_rag.extract_insights(
            transcription_text=transcription_text,
            keywords=keywords
        )
        
        # Remove raw response to avoid huge payloads
        if "raw_response" in result:
            del result["raw_response"]
        
        return result
    except Exception as e:
        logger.error(f"Error extracting insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting insights: {str(e)}")

@app.post("/api/rag/live", response_class=JSONResponse)
async def answer_from_live_transcription(
    transcription_text: str = Body(..., description="Current live transcription text"),
    query: Optional[str] = Body(None, description="Optional specific query to answer"),
    keywords: Optional[List[str]] = Body(None, description="Keywords extracted from the transcription")
):
    """Generate contextual answers from live transcription."""
    try:
        # Get LLM RAG engine
        llm_rag = get_llm_rag_engine()
        
        # Generate response
        result = llm_rag.answer_from_live_transcription(
            transcription_text=transcription_text,
            query=query,
            keywords=keywords
        )
        
        # Remove raw response to avoid huge payloads
        if "raw_response" in result:
            del result["raw_response"]
        
        return result
    except Exception as e:
        logger.error(f"Error answering from live transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering from live transcription: {str(e)}")

async def process_browser_audio(audio_chunks, call_id, source_type="browser_call", source_name="Browser"):
    """
    Process audio buffers from browser
    """
    if not audio_chunks:
        logger.warning("Empty audio buffer received, skipping processing")
        return [], []
    
    # Create directory if it doesn't exist
    os.makedirs("output_files", exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"output_files/browser_audio_{timestamp}_{call_id}.webm"
    
    try:
        # Save audio data to file
        with open(audio_filename, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)
        
        # Check if file is too small (likely silence)
        file_size = os.path.getsize(audio_filename)
        if file_size < 10 * 1024:  # Less than 10KB
            logger.info(f"Audio file too small ({file_size/1024:.1f}KB), likely silence - skipping")
            return [], []
            
        
        # SIMPLIFIED APPROACH: First try Whisper for basic transcription
        try:
            result = whisper_model.transcribe(
                temp_file,
                language="en",
                verbose=False
            )
            
            transcribed_text = result["text"].strip()
            logger.info(f"Whisper transcription: '{transcribed_text}'")
            
            # Only proceed with AssemblyAI if we have meaningful text
            if len(transcribed_text.split()) < 3:
                logger.info(f"Transcription too short, skipping AssemblyAI processing")
                return [], []
            
            # Now try AssemblyAI for speaker diarization since we know there's speech
            logger.info(f"Sending file to AssemblyAI for speaker diarization...")
            
            # Direct AssemblyAI integration
            headers = {"authorization": ASSEMBLYAI_API_KEY}
            
            # 1. Upload audio file
            with open(temp_file, 'rb') as f:
                upload_response = requests.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers=headers,
                    data=f
                )
            
            if upload_response.status_code != 200:
                logger.error(f"AssemblyAI upload error: {upload_response.text}")
                raise Exception(f"Error uploading to AssemblyAI: {upload_response.text}")
            
            upload_url = upload_response.json()["upload_url"]
            logger.info(f"Successfully uploaded audio to AssemblyAI")
            
            # 2. Request transcription with speaker diarization
            transcript_response = requests.post(
                "https://api.assemblyai.com/v2/transcript",
                headers={"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"},
                json={
                    "audio_url": upload_url,
                    "speaker_labels": True,
                    "speakers_expected": 2
                }
            )
            
            if transcript_response.status_code != 200:
                logger.error(f"AssemblyAI transcription request error: {transcript_response.text}")
                raise Exception(f"Error requesting AssemblyAI transcription: {transcript_response.text}")
            
            transcript_id = transcript_response.json()["id"]
            logger.info(f"AssemblyAI transcription requested with ID: {transcript_id}")
            
            # 3. Poll for completion
            max_polls = 30
            for i in range(max_polls):
                time.sleep(5)  # Wait 5 seconds between polls
                
                polling_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers
                )
                
                polling_data = polling_response.json()
                status = polling_data.get("status")
                
                logger.info(f"AssemblyAI status ({i+1}/{max_polls}): {status}")
                
                if status == "completed":
                    # Process the results
                    utterances = []
                    for utterance in polling_data.get("utterances", []):
                        utterances.append({
                            "speaker": utterance["speaker"],
                            "text": utterance["text"],
                            "start": utterance["start"],
                            "end": utterance["end"]
                        })
                    
                    logger.info(f"AssemblyAI returned {len(utterances)} utterances")
                    
                    # Basic speaker name detection
                    speaker_names = {}
                    for utterance in utterances:
                        speaker = utterance["speaker"]
                        text = utterance["text"].lower()
                        
                        # Simple name detection
                        name_patterns = [
                            r"(?:i am|i'm|this is|name is) (\w+)",
                            r"(\w+) (?:speaking|here)"
                        ]
                        
                        for pattern in name_patterns:
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                name = match.group(1).strip().capitalize()
                                if len(name) > 1:  # Ensure name is at least 2 characters
                                    speaker_names[speaker] = name
                                    break
                    
                    logger.info(f"Detected speaker names: {speaker_names}")
                    
                    # Extract keywords
                    full_text = " ".join([u["text"] for u in utterances])
                    keywords = AudioProcessor.extract_keywords(full_text)
                    
                    # Save to MongoDB
                    try:
                        doc = {
                            "call_id": call_id,
                            "timestamp": datetime.utcnow(),
                            "utterances": utterances,
                            "speaker_mapping": speaker_names,
                            "keywords": keywords,
                            "source_type": source_type,
                            "source_name": source_name or source_type,
                            "audio_file": temp_file
                        }
                        collection.insert_one(doc)
                        logger.info(f"Stored transcription with {len(utterances)} utterances and speakers: {speaker_names}")
                    except Exception as e:
                        logger.error(f"Error storing transcription: {str(e)}")
                    
                    return utterances, keywords
                
                elif status == "error":
                    logger.error(f"AssemblyAI error: {polling_data}")
                    break
            
            # If we reach here, either timeout or error occurred
            logger.warning("AssemblyAI processing incomplete, falling back to basic Whisper results")
            
        except Exception as e:
            logger.error(f"Error in AssemblyAI processing: {str(e)}")
            logger.info("Continuing with basic Whisper transcription")
        
        # Fallback to basic Whisper results if AssemblyAI failed
        if transcribed_text:
            # Create basic utterance from Whisper result
            utterance = {
                "speaker": source_name or "Speaker",
                "text": transcribed_text,
                "start": time.time(),
                "end": time.time()
            }
            
            # Extract keywords
            keywords = AudioProcessor.extract_keywords(transcribed_text) if len(transcribed_text.split()) > 3 else []
            
            # Save to MongoDB
            try:
                doc = {
                    "call_id": call_id,
                    "timestamp": datetime.utcnow(),
                    "utterances": [utterance],
                    "keywords": keywords,
                    "source_type": source_type,
                    "source_name": source_name or source_type,
                    "audio_file": temp_file
                }
                collection.insert_one(doc)
                logger.info(f"Stored transcription with Whisper fallback: '{transcribed_text}'")
            except Exception as e:
                logger.error(f"Error storing transcription: {str(e)}")
            
            return [utterance], keywords
        
        return [], []
        
    except Exception as e:
        logger.error(f"Error processing browser audio buffer: {str(e)}")
        traceback.print_exc()
        return [], []

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
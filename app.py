import os
import sys
import time
import json
import logging
import threading
import re
import wave
import pyaudio
import whisper
import requests
import pymongo
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

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
ASSEMBLYAI_API_KEY = "ada3df25b909471ca405dd86fc221940"  # Replace with your actual API key
RECORD_SECONDS = 30
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
OUTPUT_DIR = "recorded_audio"

# Initialize MongoDB connection
try:
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    logger.info(f"Connected to MongoDB database: {DB_NAME}, collection: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    sys.exit(1)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Whisper model
logger.info("Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL)
logger.info("Whisper model loaded!")

# Initialize KeyBERT model
logger.info("Loading KeyBERT model...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=sentence_model)
logger.info("KeyBERT model loaded!")

class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(KeywordRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Embed the input
        embeds = self.embedding(x)
        
        # Initialize hidden state
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # Apply output layer
        keyword_scores = self.sigmoid(self.fc(lstm_out))
        
        return keyword_scores, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers * 2, batch_size, self.hidden_dim).zero_(),  # *2 for bidirectional
                 weight.new(self.num_layers * 2, batch_size, self.hidden_dim).zero_())
        return hidden

class AudioSource:
    def __init__(self, source_type, source_name):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.stop_event = threading.Event()
        self.source_type = source_type  # 'microphone' or 'call'
        self.source_name = source_name
        self.utterances = []
        self.speaker_mapping = {}
        
        # Get available audio devices
        self.input_device_index = self.get_input_device()
        
    def get_input_device(self):
        """Get the appropriate input device based on source type."""
        print("\nAvailable Audio Devices:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")
            
            # For macOS, look for these device names
            if self.source_type == 'call':
                if 'BlackHole' in device_info['name'] or 'Soundflower' in device_info['name']:
                    return i
            elif self.source_type == 'microphone':
                if 'microphone' in device_info['name'].lower() or 'mic' in device_info['name'].lower():
                    return i
        
        # If no specific device found, ask user
        print(f"\nPlease select the device index for {self.source_type} recording:")
        return int(input("Enter device index: "))

    def start_recording(self):
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=CHUNK
            )
            self.is_recording = True
            self.frames = []
            
            logger.info(f"\nRecording started for {self.source_type} ({self.source_name}) using device {self.input_device_index}...")

            while not self.stop_event.is_set():
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except IOError as e:
                    logger.warning(f"Dropped frame from {self.source_type}: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error recording from {self.source_type}: {str(e)}")
        finally:
            self.stop_recording()

    def stop_recording(self):
        logger.info(f"Stopping recording for {self.source_type}...")
        self.stop_event.set()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.frames:
            self.process_recording()
        
        self.audio.terminate()

    def process_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = os.path.join(OUTPUT_DIR, f"{self.source_type}_{timestamp}.wav")
        
        # Save audio file
        with wave.open(audio_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
        
        # Process with AssemblyAI for transcription and diarization
        try:
            self.utterances = self.process_with_assemblyai(audio_filename)
            self.analyze_utterances(self.utterances)
            logger.info(f"Processing completed for {self.source_type}")
        except Exception as e:
            logger.error(f"Error processing {self.source_type} recording: {str(e)}")

    def process_with_assemblyai(self, audio_file):
        """Process audio file with AssemblyAI for speaker diarization."""
        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json"  # For API requests
        }
        
        # Upload file
        logger.info(f"Sending {self.source_type} file to AssemblyAI...")
        upload_url = None
        
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
            logger.info(f"{self.source_type} file uploaded successfully")
            
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
            logger.info(f"{self.source_type} transcription started with ID: {transcript_id}")
            
            # Wait for completion
            while True:
                polling_response = requests.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers
                )
                
                polling_data = polling_response.json()
                status = polling_data["status"]
                
                if status == "completed":
                    logger.info(f"{self.source_type} transcription completed successfully")
                    break
                elif status == "error":
                    raise Exception(f"Transcription error: {polling_data}")
                
                print(".", end="", flush=True)
                time.sleep(3)
            
            print(f"\n{self.source_type} processing completed!")
            
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
                logger.warning(f"No speaker-separated utterances found for {self.source_type}")
                utterances = [{
                    "speaker": "A",
                    "text": polling_data.get("text", ""),
                    "start": 0,
                    "end": 0
                }]
            
            return utterances
            
        except Exception as e:
            logger.error(f"AssemblyAI API error for {self.source_type}: {str(e)}")
            raise

    def analyze_utterances(self, utterances):
        """Analyze utterances and identify speakers from conversation."""
        # Dictionary to store potential names mentioned for each speaker
        speaker_mentions = {}
        
        # First pass: Find all potential speaker names
        for utterance in utterances:
            speaker_id = utterance["speaker"]
            text = utterance["text"].lower()
            
            # Initialize speaker mentions if not exists
            if speaker_id not in speaker_mentions:
                speaker_mentions[speaker_id] = set()
            
            # Look for introduction patterns
            intro_patterns = [
                r"(?:i am|i'm|this is|my name is) (\w+)",
                r"(?:i'm|i am) (?:called) (\w+)",
                r"(\w+) (?:speaking|here)",
                r"(?:this is) (\w+) (?:speaking)",
                r"(?:^|\s)i(?:'m|\sam)\s+(\w+)",  # Catches "I am Mark" more reliably
                r"hello[,.]?\s+(?:i(?:'m|\sam))?\s*(\w+)(?:\s+(?:here|speaking))?"  # Catches "Hello, Mark here"
            ]
            
            for pattern in intro_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match.group(1).strip().capitalize()
                    if len(name) > 1 and name.lower() not in ['am', 'here', 'speaking']:  # Filter common false positives
                        speaker_mentions[speaker_id].add(name)
        
        # Second pass: Assign names to speakers
        for speaker_id, names in speaker_mentions.items():
            if names:
                # If multiple names found for a speaker, use the first one
                self.speaker_mapping[speaker_id] = list(names)[0]
            else:
                # If no name found, keep the default speaker ID
                self.speaker_mapping[speaker_id] = f"Speaker {speaker_id}"

class ConversationManager:
    def __init__(self):
        print("\nInitializing audio sources...")
        print("First, let's set up the microphone recording.")
        self.mic_source = AudioSource('microphone', 'Local Speaker')
        
        print("\nNow, let's set up the call audio recording.")
        print("NOTE: To record call audio, you need an audio routing tool installed:")
        print("- For macOS: Install 'BlackHole' or 'Soundflower'")
        print("- For Windows: Use 'Virtual Audio Cable'")
        print("- For Linux: Use 'PulseAudio' or 'JACK'")
        print("\nMake sure your system audio is routed to the virtual audio device.")
        
        self.call_source = AudioSource('call', 'Remote Speaker')
        self.merged_utterances = []
        self.stop_event = threading.Event()
        
        # Initialize tokenizer for RNN
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Initialize RNN model (would need to be trained)
        vocab_size = self.tokenizer.vocab_size
        self.rnn_model = KeywordRNN(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=2
        )

    def extract_keywords_rnn(self, text, top_n=3):
        """Extract keywords using RNN-based approach."""
        try:
            # Tokenize input text
            tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get word importance scores from RNN
            with torch.no_grad():
                scores, _ = self.rnn_model(tokens['input_ids'])
                scores = scores.squeeze()
            
            # Convert tokens back to words and pair with scores
            words = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
            word_scores = [(word, float(score)) for word, score in zip(words, scores) 
                          if not word.startswith('##') and word not in ['[CLS]', '[SEP]', '[PAD]']]
            
            # Sort by score and get top_n
            word_scores.sort(key=lambda x: x[1], reverse=True)
            return word_scores[:top_n]
            
        except Exception as e:
            logger.error(f"Error extracting keywords with RNN: {str(e)}")
            return []

    def extract_keywords(self, text, top_n=3, method='keybert'):
        """Extract keywords using specified method."""
        if method == 'rnn':
            return self.extract_keywords_rnn(text, top_n)
        else:
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
                logger.error(f"Error extracting keywords with KeyBERT: {str(e)}")
                return []

    def start_recording(self):
        # Create threads for each audio source
        mic_thread = threading.Thread(target=self.mic_source.start_recording)
        call_thread = threading.Thread(target=self.call_source.start_recording)

        logger.info("Starting recording from both sources...")
        mic_thread.start()
        call_thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(1)  # Check for stop signal every second
        except KeyboardInterrupt:
            logger.info("Stopping recording...")
            self.stop_recording()
            mic_thread.join()
            call_thread.join()
            self.merge_conversations()

    def stop_recording(self):
        self.stop_event.set()
        self.mic_source.stop_event.set()
        self.call_source.stop_event.set()

    def merge_conversations(self):
        # Combine utterances from both sources and sort by timestamp
        all_utterances = []
        
        # Add source information to utterances and extract keywords
        for utterance in self.mic_source.utterances:
            utterance['source'] = 'microphone'
            # Extract keywords for this utterance
            keywords = self.extract_keywords(utterance['text'])
            utterance['keywords'] = keywords
            all_utterances.append(utterance)
        
        for utterance in self.call_source.utterances:
            utterance['source'] = 'call'
            # Extract keywords for this utterance
            keywords = self.extract_keywords(utterance['text'])
            utterance['keywords'] = keywords
            all_utterances.append(utterance)
        
        # Sort by start time
        self.merged_utterances = sorted(all_utterances, key=lambda x: x['start'])
        
        # Save merged conversation
        self.save_merged_conversation()

    def save_merged_conversation(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = os.path.join(OUTPUT_DIR, f"merged_conversation_{timestamp}.txt")
        
        with open(merged_file, 'w') as f:
            f.write("=== Merged Conversation ===\n\n")
            for utterance in self.merged_utterances:
                source = utterance['source']
                speaker_name = (self.mic_source.speaker_mapping.get(utterance['speaker']) 
                              if source == 'microphone' 
                              else self.call_source.speaker_mapping.get(utterance['speaker']))
                speaker_name = speaker_name or f"Speaker {utterance['speaker']}"
                
                # Format keywords as a string
                keywords_str = ', '.join([f"{word} ({score:.2f})" for word, score in utterance.get('keywords', [])])
                
                f.write(f"[{source}] {speaker_name}: {utterance['text']}\n")
                if keywords_str:
                    f.write(f"Keywords: {keywords_str}\n")
                f.write("\n")
        
        logger.info(f"Merged conversation saved to: {merged_file}")
        
        # Save to MongoDB
        data = {
            "timestamp": datetime.now(),
            "utterances": self.merged_utterances,
            "mic_speaker_mapping": self.mic_source.speaker_mapping,
            "call_speaker_mapping": self.call_source.speaker_mapping
        }
        result = collection.insert_one(data)
        logger.info(f"Conversation saved to MongoDB with ID: {result.inserted_id}")

def manual_speaker_assignment(audio_file):
    """Manually assign speaker names before processing."""
    speaker_mapping = {}
    print("\nManual speaker assignment:")
    
    num_speakers = int(input("How many different speakers are in the recording? "))
    
    for i in range(1, num_speakers + 1):
        speaker_id = f"Speaker {chr(64 + i)}"  # A, B, C, etc.
        name = input(f"Enter name for {speaker_id}: ").strip().capitalize()
        if name:
            speaker_mapping[speaker_id] = name
    
    return speaker_mapping

def process_existing_file(audio_file, speaker_mapping=None):
    """Process an existing audio file."""
    recorder = AudioRecorder()
    
    if speaker_mapping:
        recorder.speaker_mapping = speaker_mapping
    
    # Read audio file
    with wave.open(audio_file, 'rb') as wf:
        frames = []
        data = wf.readframes(wf.getnframes())
        frames.append(data)
    
    recorder.process_recording()

def main():
    print("Multi-Source Audio Transcription System")
    print("======================================")
    
    manager = ConversationManager()
    logger.info(f"Initializing... Files will be saved to {os.path.abspath(OUTPUT_DIR)}")
    logger.info("Recording from both microphone and call...")
    logger.info("Press Ctrl+C to stop recording")
    
    manager.start_recording()

if __name__ == "__main__":
    main()
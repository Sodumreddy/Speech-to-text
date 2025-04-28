// microphone-capture.js
// Handles microphone capture and streaming to the server via WebSocket

let audioContext = null;
let microphoneStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let websocket = null;
let callId = null;

// Start capturing audio from the microphone
async function startMicrophoneCapture() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000,
                channelCount: 1
            }
        });
        
        // Save the stream for later cleanup
        microphoneStream = stream;
        
        // Generate a unique call ID for this recording
        callId = 'mic_' + Date.now();
        console.log(`Starting microphone capture with call ID: ${callId}`);
        
        // Create WebSocket connection
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/transcription`;
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = function() {
            console.log("WebSocket connected for microphone");
            
            // Send start message with source information
            websocket.send(JSON.stringify({
                type: "start",
                config: {
                    source_type: "microphone",
                    source_name: "Browser Microphone",
                    use_llm: true,
                    extract_keywords: true
                }
            }));
            
            // Set up MediaRecorder once WebSocket is connected
            setupMediaRecorder(stream);
        };
        
        websocket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                
                // Handle different message types
                if (data.type === "transcription") {
                    // This is handled by the main code in index.html
                    const transcriptionEvent = new CustomEvent('transcription-received', { 
                        detail: data 
                    });
                    document.dispatchEvent(transcriptionEvent);
                }
                else if (data.type === "session_started") {
                    callId = data.call_id;
                    console.log(`Microphone session started with ID: ${callId}`);
                }
            } catch (e) {
                console.error("Error parsing WebSocket message:", e);
            }
        };
        
        websocket.onclose = function() {
            console.log("WebSocket closed for microphone capture");
            stopMicrophoneCapture();
        };
        
        websocket.onerror = function(error) {
            console.error("WebSocket error:", error);
            document.getElementById('status').textContent = "WebSocket error";
        };
        
    } catch (error) {
        console.error("Error starting microphone capture:", error);
        document.getElementById('status').textContent = `Error: ${error.message}`;
    }
}

// Set up MediaRecorder to capture audio chunks
function setupMediaRecorder(stream) {
    // Try different MIME types for better browser compatibility
    let mimeType = 'audio/webm;codecs=opus';
    
    // Check if the browser supports this MIME type
    if (!MediaRecorder.isTypeSupported(mimeType)) {
        // Try alternative MIME types in order of preference
        const alternatives = [
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg',
            ''  // Empty string = browser default
        ];
        
        for (let alt of alternatives) {
            if (MediaRecorder.isTypeSupported(alt)) {
                mimeType = alt;
                console.log(`Using alternative MIME type: ${mimeType}`);
                break;
            }
        }
    }
    
    // Set options for MediaRecorder
    const options = {
        audioBitsPerSecond: 16000
    };
    
    // Only set mimeType if it's not empty (let browser choose default if none supported)
    if (mimeType) {
        options.mimeType = mimeType;
    }
    
    console.log(`Creating MediaRecorder with options:`, options);
    
    try {
        // Create MediaRecorder
        mediaRecorder = new MediaRecorder(stream, options);
        
        // Handle data available event
        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                // Send audio data directly over WebSocket
                websocket.send(event.data);
                console.log(`Sent audio chunk: ${event.data.size} bytes`);
            }
        };
        
        // Start recording, getting data every 500ms
        mediaRecorder.start(500);
        console.log("MediaRecorder started successfully");
        
        // Update status
        document.getElementById('status').textContent = "Recording from microphone...";
    } catch (error) {
        console.error("MediaRecorder error:", error);
        document.getElementById('status').textContent = `MediaRecorder error: ${error.message}`;
    }
}

// Stop microphone capture
function stopMicrophoneCapture() {
    // Stop MediaRecorder if active
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        console.log("MediaRecorder stopped");
    }
    
    // Stop all audio tracks
    if (microphoneStream) {
        microphoneStream.getTracks().forEach(track => track.stop());
        microphoneStream = null;
        console.log("Microphone tracks stopped");
    }
    
    // Close WebSocket connection
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: 'stop' }));
        setTimeout(() => {
            websocket.close();
            websocket = null;
        }, 500);
    }
    
    console.log("Microphone capture stopped");
} 
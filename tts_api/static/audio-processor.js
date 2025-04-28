// audio-processor.js
// This is an AudioWorkletProcessor that processes audio data from the microphone
// and passes it to the main thread for streaming to the server.

class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 4096; // Buffer size for audio data
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
    
    // We'll use a simple voice activity detection
    this.silenceThreshold = 0.01;
    this.isSpeaking = false;
    this.silenceCounter = 0;
    this.requiredSilenceFrames = 30; // About 0.6 seconds of silence
  }

  process(inputs, outputs, parameters) {
    // Get the input data (microphone)
    const input = inputs[0];
    if (!input || !input.length) return true;

    const channelData = input[0]; // Use first channel (mono)
    
    // Check if this frame contains speech
    let maxAmplitude = 0;
    for (let i = 0; i < channelData.length; i++) {
      if (Math.abs(channelData[i]) > maxAmplitude) {
        maxAmplitude = Math.abs(channelData[i]);
      }
    }
    
    const isSilent = maxAmplitude < this.silenceThreshold;
    
    if (!isSilent) {
      // Reset silence counter when speech is detected
      this.silenceCounter = 0;
      
      if (!this.isSpeaking) {
        // Transition from silence to speech
        this.isSpeaking = true;
        console.log("Speech detected");
      }
    } else {
      // Increment silence counter
      this.silenceCounter++;
      
      if (this.isSpeaking && this.silenceCounter > this.requiredSilenceFrames) {
        // Transition from speech to silence
        this.isSpeaking = false;
        console.log("Speech ended");
        
        // Send the buffer when speech ends
        if (this.bufferIndex > 0) {
          this.sendBufferToMainThread();
          this.bufferIndex = 0;
        }
      }
    }
    
    // Append the new data to our buffer with gain boost
    for (let i = 0; i < channelData.length; i++) {
      // Apply a gain boost to increase volume
      const boostedSample = channelData[i] * 2.5;
      // Clip to prevent distortion
      this.buffer[this.bufferIndex++] = Math.max(-1.0, Math.min(1.0, boostedSample));
      
      // If our buffer is full, send it and reset
      if (this.bufferIndex >= this.bufferSize) {
        this.sendBufferToMainThread();
        this.bufferIndex = 0;
      }
    }
    
    // Keep the processor alive
    return true;
  }
  
  sendBufferToMainThread() {
    // Convert the float audio data to 16-bit PCM
    const pcmData = this.convertToPCM(this.buffer, this.bufferIndex);
    
    // Send the PCM data to the main thread
    this.port.postMessage({
      audio: pcmData,
      isSpeaking: this.isSpeaking
    });
  }
  
  convertToPCM(float32Array, length) {
    // Convert Float32Array to Int16Array (16-bit PCM)
    const int16Array = new Int16Array(length);
    
    for (let i = 0; i < length; i++) {
      // Convert float value (-1.0 to 1.0) to int16 (-32768 to 32767)
      let sample = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    }
    
    return int16Array.buffer;
  }
}

// Register the processor
registerProcessor('audio-processor', AudioProcessor); 
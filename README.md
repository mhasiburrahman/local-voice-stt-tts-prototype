# Local-Only Real-Time STT/TTS Prototype

## Project Overview
A CPU-compatible, local speech interface prototype developed for the Cybernetic Human Lab Virtual Human project.

The prototype captures microphone audio, detects speech, transcribes it, generates a simple text response, converts the response into speech, and plays it automatically.

## System Architecture
Microphone -> Silero VAD -> faster-whisper -> Dialogue Interface -> Piper TTS -> Speaker

## Technologies
- Python
- Silero VAD: Speech detection
- faster-whisper (tiny.en): Speech-to-text
- Piper TTS (en_GB-alan-medium): Text-to-speech
- SoundDevice: Audio capture and playback
- PyTorch: CPU-based VAD inference

## Key Features
- Automatic speech detection and endpointing
- Local speech recognition
- Local speech synthesis
- Persistent Piper model loading
- Automatic audio playback
- Continuous listening between turns
- STT and TTS latency measurements
- Real-Time Factor (RTF) measurements
- CPU-only operation

## Hardware Tested
- Intel Core i5-10210U
- 8 GB RAM
- Intel integrated graphics
- Windows / PowerShell
- No NVIDIA GPU required

## Installation
Python 3.11 is recommended.

Install the dependencies:

    pip install -r requirements.txt

Download the Piper voice model `en_GB-alan-medium` and its configuration file into the `voices` directory.

The application expects:

    voices/en_GB-alan-medium.onnx
    voices/en_GB-alan-medium.onnx.json

Set `MIC_DEVICE` in `local_voice_system.py` to the appropriate microphone device index for your computer.

## Running the Prototype

    python local_voice_system.py

Speak into the microphone. The system will detect speech, transcribe it, generate a response, synthesize speech and return to listening.

Press Ctrl+C to stop.

## Latest Prototype Results

| Metric | Result |
|---|---|
| Captured speech | 1.63 s |
| STT processing | 0.91 s |
| STT RTF | 0.56 |
| TTS generation | 0.74 s |
| TTS RTF | 0.34 |
| Processing only | 1.65 s |
| Processing + full playback | 4.29 s |

Results are from one successful demonstration turn, not averages.

## Experimental Evaluation
The project also evaluated Whisper-Loop, Whisper Local, and Piper TTS.

A controlled 10-utterance STT benchmark compared faster-whisper tiny.en and base.en.

| Metric | tiny.en | base.en |
|---|---|---|
| Average WER | 9.66% | 11.42% |
| Average latency | 0.537 s | 1.020 s |
| Average RTF | 0.122 | 0.239 |

These findings are specific to the tested audio samples and hardware.

## Current Limitations
- The dialogue layer is currently rule-based.
- Incremental streaming STT is not implemented.
- Virtual Human integration is pending.
- WebSocket/API communication is not yet implemented.
- Full CPU/RAM profiling and VAD latency measurement remain future work.

## Future Development
- Integrate the Virtual Human dialogue system
- Implement a streaming communication interface
- Improve latency monitoring
- Extend performance and resource benchmarking

## Third-Party Technologies
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Silero VAD: https://github.com/snakers4/silero-vad
- Piper: https://github.com/rhasspy/piper
- Whisper-Loop (evaluated): https://github.com/aryeo0908/whisper-loop

## Project Status
Working local STT/TTS proof of concept. Further integration and performance evaluation are planned.

import sounddevice as sd
import numpy as np
import torch
import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
from piper import PiperVoice


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
MIC_DEVICE = 1
CHUNK_SIZE = 512

SPEECH_THRESHOLD = 0.5
SILENCE_DURATION = 1.0
MAX_RECORDING_SECONDS = 30

WHISPER_MODEL = "tiny.en"

PIPER_VOICE_PATH = Path("voices/en_GB-alan-medium.onnx")
TTS_OUTPUT_FILE = "response.wav"


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("LOCAL REAL-TIME VOICE PROTOTYPE")
print("=" * 60)

print("\n[1/3] Loading Silero VAD...")
vad_model = load_silero_vad()

print("[2/3] Loading faster-whisper...")
whisper_model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

print("[3/3] Loading Piper TTS...")

piper_load_start = time.perf_counter()

piper_voice = PiperVoice.load(
    str(PIPER_VOICE_PATH)
)

piper_load_time = time.perf_counter() - piper_load_start

print(
    f"Piper loaded in {piper_load_time:.2f} seconds."
)

print("\nAll models loaded successfully.")


# ============================================================
# SIMPLE RESPONSE / DIALOGUE INTERFACE
# ============================================================

def generate_response(user_text):

    text = user_text.lower()

    if "hello" in text or "hi" in text:
        return "Hello! Nice to hear from you."

    elif "how are you" in text:
        return "I am working perfectly. Thank you for asking."

    elif "your name" in text:
        return "I am your local virtual human voice system."

    elif "what are you" in text:
        return (
            "I am a completely local speech recognition "
            "and text to speech prototype."
        )

    elif "good morning" in text:
        return "Good morning! I hope you are having a good day."

    elif "thank you" in text or "thanks" in text:
        return "You're welcome."

    elif "goodbye" in text or "bye" in text:
        return "Goodbye. It was nice talking to you."

    else:
        return f"I heard you say: {user_text}"


# ============================================================
# PERSISTENT PIPER TTS
# ============================================================

def speak(text):

    print(f"\nSYSTEM: {text}")
    print("[TTS] Generating speech...")

    start = time.perf_counter()

    # Piper is already loaded in memory.
    # Only synthesis happens here.
    with wave.open(TTS_OUTPUT_FILE, "wb") as wav_file:

        piper_voice.synthesize_wav(
            text,
            wav_file
        )

    generation_time = time.perf_counter() - start

    print(
        f"[TTS] Generation time: "
        f"{generation_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # READ GENERATED AUDIO
    # --------------------------------------------------------

    with wave.open(TTS_OUTPUT_FILE, "rb") as wav_file:

        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()

        audio_data = wav_file.readframes(frames)

    audio_duration = frames / float(sample_rate)

    if sample_width == 2:

        audio = np.frombuffer(
            audio_data,
            dtype=np.int16
        )

    else:

        raise ValueError(
            f"Unsupported WAV sample width: {sample_width}"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        )

    # Calculate TTS RTF
    if audio_duration > 0:

        tts_rtf = (
            generation_time / audio_duration
        )

    else:

        tts_rtf = 0

    print(
        f"[TTS] Audio duration: "
        f"{audio_duration:.2f} seconds"
    )

    print(
        f"[TTS] RTF: "
        f"{tts_rtf:.2f}"
    )

    # --------------------------------------------------------
    # PLAY AUDIO
    # --------------------------------------------------------

    print("[TTS] Speaking...")

    playback_start = time.perf_counter()

    sd.play(
        audio,
        samplerate=sample_rate
    )

    # Wait until speech finishes before
    # microphone listening resumes.
    sd.wait()

    playback_time = (
        time.perf_counter() - playback_start
    )

    return (
        generation_time,
        playback_time,
        audio_duration,
        tts_rtf
    )


# ============================================================
# SPEECH-TO-TEXT
# ============================================================

def transcribe_audio(audio_chunks):

    audio = np.concatenate(
        audio_chunks
    ).astype(np.float32)

    audio_duration = (
        len(audio) / SAMPLE_RATE
    )

    print(
        f"\n[STT] Speech captured: "
        f"{audio_duration:.2f} seconds"
    )

    print("[STT] Transcribing...")

    start = time.perf_counter()

    segments, info = whisper_model.transcribe(
        audio,
        language="en",
        vad_filter=False
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    stt_time = (
        time.perf_counter() - start
    )

    if audio_duration > 0:

        stt_rtf = (
            stt_time / audio_duration
        )

    else:

        stt_rtf = 0

    print("\n" + "-" * 60)
    print(f"YOU: {text}")
    print("-" * 60)

    print(
        f"[STT] Processing time: "
        f"{stt_time:.2f} seconds"
    )

    print(
        f"[STT] RTF: "
        f"{stt_rtf:.2f}"
    )

    return (
        text,
        stt_time,
        stt_rtf,
        audio_duration
    )


# ============================================================
# MAIN LOOP
# ============================================================

print("\n" + "=" * 60)
print("SYSTEM READY")
print("=" * 60)

print("\nArchitecture:")
print(
    "Microphone -> Silero VAD -> faster-whisper "
    "-> Dialogue -> Piper -> Speaker"
)

print("\nStart speaking whenever you're ready.")
print("Press Ctrl+C to stop.\n")


try:

    while True:

        audio_buffer = []

        is_speaking = False

        silence_start = None
        recording_start = None

        print("[VAD] Listening...")

        # ====================================================
        # MICROPHONE + VAD
        # ====================================================

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            device=MIC_DEVICE
        ) as stream:

            while True:

                audio_chunk, overflowed = stream.read(
                    CHUNK_SIZE
                )

                chunk = audio_chunk[:, 0].copy()

                tensor = torch.from_numpy(
                    chunk
                )

                speech_probability = vad_model(
                    tensor,
                    SAMPLE_RATE
                ).item()

                # --------------------------------------------
                # SPEECH DETECTED
                # --------------------------------------------

                if (
                    speech_probability
                    >= SPEECH_THRESHOLD
                ):

                    if not is_speaking:

                        print(
                            "[VAD] Speech detected."
                        )

                        is_speaking = True

                        audio_buffer = []

                        recording_start = (
                            time.time()
                        )

                    audio_buffer.append(
                        chunk
                    )

                    silence_start = None

                # --------------------------------------------
                # SILENCE
                # --------------------------------------------

                elif is_speaking:

                    audio_buffer.append(
                        chunk
                    )

                    if silence_start is None:

                        silence_start = (
                            time.time()
                        )

                    silence_time = (
                        time.time()
                        - silence_start
                    )

                    recording_time = (
                        time.time()
                        - recording_start
                    )

                    if (
                        silence_time
                        >= SILENCE_DURATION
                        or
                        recording_time
                        >= MAX_RECORDING_SECONDS
                    ):

                        print(
                            "[VAD] End of speech detected."
                        )

                        break

        # Microphone is now closed before
        # STT/TTS processing.

        if not audio_buffer:
            continue

        # ====================================================
        # START TURN TIMER
        # ====================================================

        processing_start = (
            time.perf_counter()
        )

        # ====================================================
        # STT
        # ====================================================

        (
            user_text,
            stt_time,
            stt_rtf,
            captured_audio_duration
        ) = transcribe_audio(
            audio_buffer
        )

        if not user_text.strip():

            print(
                "\nNo transcription detected.\n"
            )

            continue

        # ====================================================
        # DIALOGUE
        # ====================================================

        response_start = (
            time.perf_counter()
        )

        response = generate_response(
            user_text
        )

        response_time = (
            time.perf_counter()
            - response_start
        )

        # ====================================================
        # TTS
        # ====================================================

        (
            tts_time,
            playback_time,
            generated_audio_duration,
            tts_rtf
        ) = speak(
            response
        )

        total_time = (
            time.perf_counter()
            - processing_start
        )

        processing_only = (
            stt_time
            + response_time
            + tts_time
        )

        # ====================================================
        # TURN METRICS
        # ====================================================

        print("\n" + "=" * 60)
        print("TURN METRICS")
        print("=" * 60)

        print(
            f"Captured speech:       "
            f"{captured_audio_duration:.2f} s"
        )

        print(
            f"STT processing:        "
            f"{stt_time:.2f} s"
        )

        print(
            f"STT RTF:               "
            f"{stt_rtf:.2f}"
        )

        print(
            f"Dialogue processing:   "
            f"{response_time:.4f} s"
        )

        print(
            f"TTS generation:        "
            f"{tts_time:.2f} s"
        )

        print(
            f"TTS RTF:               "
            f"{tts_rtf:.2f}"
        )

        print(
            f"Generated speech:      "
            f"{generated_audio_duration:.2f} s"
        )

        print(
            f"Processing only:       "
            f"{processing_only:.2f} s"
        )

        print(
            f"Playback duration:     "
            f"{playback_time:.2f} s"
        )

        print(
            f"Processing + playback: "
            f"{total_time:.2f} s"
        )

        print("=" * 60)
        print()


except KeyboardInterrupt:

    print(
        "\n\nLocal voice prototype stopped."
    )


except Exception as e:

    print(
        f"\nERROR: {e}"
    )
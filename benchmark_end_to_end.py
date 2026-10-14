import sounddevice as sd
import numpy as np
import torch
import time
import csv
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
PIPER_MODEL = Path("voices/en_GB-alan-medium.onnx")

NUMBER_OF_TESTS = 5

OUTPUT_CSV = "end_to_end_results.csv"


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("=" * 65)
print("LOCAL VOICE SYSTEM - END-TO-END BENCHMARK")
print("=" * 65)

print("\nLoading Silero VAD...")
vad_model = load_silero_vad()

print("Loading Whisper...")
whisper_model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

print("Loading Piper...")
piper_voice = PiperVoice.load(
    str(PIPER_MODEL)
)

print("All models loaded.\n")


# ============================================================
# WARM-UP
# ============================================================

print("Warming up Piper...")

for _ in piper_voice.synthesize(
    "System ready."
):
    pass

print("Warm-up complete.\n")


# ============================================================
# RESPONSE LOGIC
# ============================================================

def generate_response(text):

    text_lower = text.lower()

    if "hello" in text_lower:
        return "Hello. Nice to hear from you."

    if "how are you" in text_lower:
        return "I am working well. Thank you."

    if "name" in text_lower:
        return "I am your local virtual human system."

    if "morning" in text_lower:
        return "Good morning. I hope you are having a good day."

    if "thank" in text_lower:
        return "You're welcome."

    return "I heard what you said."


# ============================================================
# LISTEN FOR ONE UTTERANCE
# ============================================================

def listen():

    audio_buffer = []
    speaking = False

    silence_start = None
    recording_start = None

    print("Listening...")

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

            probability = vad_model(
                torch.from_numpy(chunk),
                SAMPLE_RATE
            ).item()

            if probability >= SPEECH_THRESHOLD:

                if not speaking:

                    print("Speech detected.")

                    speaking = True
                    recording_start = time.perf_counter()

                audio_buffer.append(chunk)

                silence_start = None

            elif speaking:

                audio_buffer.append(chunk)

                if silence_start is None:

                    # Approximate beginning of final silence
                    silence_start = time.perf_counter()

                silence_duration = (
                    time.perf_counter()
                    - silence_start
                )

                recording_duration = (
                    time.perf_counter()
                    - recording_start
                )

                if (
                    silence_duration >= SILENCE_DURATION
                    or
                    recording_duration >= MAX_RECORDING_SECONDS
                ):

                    endpoint_detected = time.perf_counter()

                    print("End of speech detected.")

                    break

    audio = np.concatenate(
        audio_buffer
    ).astype(np.float32)

    return audio, endpoint_detected


# ============================================================
# BENCHMARK
# ============================================================

results = []

for test_number in range(
    1,
    NUMBER_OF_TESTS + 1
):

    print("\n")
    print("=" * 65)
    print(
        f"TEST {test_number}/{NUMBER_OF_TESTS}"
    )
    print("=" * 65)

    print(
        "Speak a normal sentence when ready."
    )

    audio, endpoint_time = listen()

    # --------------------------------------------------------
    # STT
    # --------------------------------------------------------

    stt_start = time.perf_counter()

    segments, info = whisper_model.transcribe(
        audio,
        language="en",
        vad_filter=False
    )

    user_text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    stt_end = time.perf_counter()

    stt_latency = (
        stt_end - stt_start
    )

    print(f"\nYOU: {user_text}")
    print(
        f"STT latency: "
        f"{stt_latency:.3f} s"
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    response_start = time.perf_counter()

    response = generate_response(
        user_text
    )

    response_end = time.perf_counter()

    response_latency = (
        response_end - response_start
    )

    print(f"SYSTEM: {response}")

    # --------------------------------------------------------
    # TTS + FIRST AUDIO CHUNK
    # --------------------------------------------------------

    tts_start = time.perf_counter()

    first_audio_time = None
    audio_chunks = []

    for chunk in piper_voice.synthesize(
        response
    ):

        if first_audio_time is None:
            first_audio_time = (
                time.perf_counter()
            )

        audio_chunks.append(chunk)

    tts_end = time.perf_counter()

    ttfa = (
        first_audio_time - tts_start
    )

    total_tts = (
        tts_end - tts_start
    )

    # --------------------------------------------------------
    # END-TO-END
    # --------------------------------------------------------

    endpoint_to_first_audio = (
        first_audio_time - endpoint_time
    )

    print("\nMETRICS")
    print("-" * 45)

    print(
        f"STT latency:              "
        f"{stt_latency:.3f} s"
    )

    print(
        f"Response logic latency:   "
        f"{response_latency:.6f} s"
    )

    print(
        f"TTS TTFA:                 "
        f"{ttfa:.3f} s"
    )

    print(
        f"TTS total generation:     "
        f"{total_tts:.3f} s"
    )

    print(
        f"Endpoint -> first audio:  "
        f"{endpoint_to_first_audio:.3f} s"
    )

    results.append({
        "test": test_number,
        "transcription": user_text,
        "response": response,
        "stt_latency_seconds": round(
            stt_latency, 3
        ),
        "response_latency_seconds": round(
            response_latency, 6
        ),
        "tts_ttfa_seconds": round(
            ttfa, 3
        ),
        "tts_total_seconds": round(
            total_tts, 3
        ),
        "endpoint_to_first_audio_seconds": round(
            endpoint_to_first_audio, 3
        )
    })

    print(
        "\nTest complete. "
        "Prepare for the next test."
    )

    time.sleep(1)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    fieldnames = [
        "test",
        "transcription",
        "response",
        "stt_latency_seconds",
        "response_latency_seconds",
        "tts_ttfa_seconds",
        "tts_total_seconds",
        "endpoint_to_first_audio_seconds"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

avg_stt = np.mean([
    r["stt_latency_seconds"]
    for r in results
])

avg_ttfa = np.mean([
    r["tts_ttfa_seconds"]
    for r in results
])

avg_e2e = np.mean([
    r["endpoint_to_first_audio_seconds"]
    for r in results
])


print("\n")
print("=" * 65)
print("END-TO-END BENCHMARK SUMMARY")
print("=" * 65)

print(
    f"Average STT latency:             "
    f"{avg_stt:.3f} s"
)

print(
    f"Average TTS TTFA:                "
    f"{avg_ttfa:.3f} s"
)

print(
    f"Average endpoint-to-first-audio: "
    f"{avg_e2e:.3f} s"
)

print("=" * 65)

print(
    f"\nResults saved to: {OUTPUT_CSV}"
)
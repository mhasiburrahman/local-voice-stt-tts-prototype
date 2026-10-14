import sounddevice as sd
import numpy as np
import time
import psutil
import os

from faster_whisper import WhisperModel
from jiwer import wer


SAMPLE_RATE = 16000
MIC_DEVICE = 1
DURATION = 8

MODELS = [
    "tiny.en",
    "base.en"
]


print("=" * 60)
print("LOCAL STT BENCHMARK")
print("=" * 60)

# ------------------------------------------------------------
# Reference sentence
# ------------------------------------------------------------

reference = input(
    "\nType the EXACT sentence you are going to speak:\n> "
).strip()

print(
    f"\nYou will have {DURATION} seconds to speak."
)

input("Press ENTER when ready...")

print("\n🎤 Speak now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=MIC_DEVICE
)

sd.wait()

audio = audio.flatten()

print("Recording complete.\n")

# Remove near-silent beginning/end for fairer RTF
threshold = 0.01

active = np.where(
    np.abs(audio) > threshold
)[0]

if len(active) > 0:
    audio = audio[
        active[0]:active[-1] + 1
    ]

audio_duration = len(audio) / SAMPLE_RATE

print(
    f"Audio duration used: "
    f"{audio_duration:.2f} seconds"
)

print("\nStarting benchmark...\n")


results = []

for model_name in MODELS:

    print("=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    print("Loading model...")

    load_start = time.perf_counter()

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8"
    )

    load_time = (
        time.perf_counter() - load_start
    )

    process = psutil.Process(
        os.getpid()
    )

    ram_before = (
        process.memory_info().rss
        / (1024 ** 2)
    )

    cpu_before = (
        process.cpu_times().user
        + process.cpu_times().system
    )

    start = time.perf_counter()

    segments, info = model.transcribe(
        audio,
        language="en",
        vad_filter=False
    )

    transcription = " ".join(
        segment.text.strip()
        for segment in segments
    )

    processing_time = (
        time.perf_counter() - start
    )

    cpu_after = (
        process.cpu_times().user
        + process.cpu_times().system
    )

    ram_after = (
        process.memory_info().rss
        / (1024 ** 2)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    error_rate = wer(
        reference.lower(),
        transcription.lower()
    )

    rtf = (
        processing_time / audio_duration
        if audio_duration > 0
        else 0
    )

    cpu_time = cpu_after - cpu_before

    ram_change = ram_after - ram_before

    print(f"\nReference:")
    print(reference)

    print(f"\nTranscription:")
    print(transcription)

    print("\nRESULTS")
    print("-" * 40)

    print(
        f"Model load time: "
        f"{load_time:.2f} s"
    )

    print(
        f"STT latency: "
        f"{processing_time:.2f} s"
    )

    print(
        f"RTF: "
        f"{rtf:.3f}"
    )

    print(
        f"WER: "
        f"{error_rate * 100:.2f}%"
    )

    print(
        f"Process RAM after STT: "
        f"{ram_after:.1f} MB"
    )

    print(
        f"RAM change during STT: "
        f"{ram_change:.1f} MB"
    )

    print(
        f"CPU time consumed: "
        f"{cpu_time:.2f} s"
    )

    results.append({
        "model": model_name,
        "transcription": transcription,
        "latency": processing_time,
        "rtf": rtf,
        "wer": error_rate * 100,
        "ram": ram_after
    })

    del model

    print()


# ------------------------------------------------------------
# Comparison
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("FINAL COMPARISON")
print("=" * 75)

print(
    f"{'Model':<12}"
    f"{'Latency':<12}"
    f"{'RTF':<10}"
    f"{'WER':<12}"
    f"{'RAM':<12}"
)

print("-" * 75)

for result in results:

    print(
        f"{result['model']:<12}"
        f"{result['latency']:<12.2f}"
        f"{result['rtf']:<10.3f}"
        f"{result['wer']:<12.2f}"
        f"{result['ram']:<12.1f}"
    )

print("=" * 75)

print("\nBenchmark completed.")
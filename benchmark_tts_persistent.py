import time
import wave
import csv
import numpy as np
from pathlib import Path
from piper import PiperVoice

VOICE_PATH = Path("voices/en_GB-alan-medium.onnx")
OUTPUT_CSV = "tts_persistent_results.csv"

TESTS = [
    ("TTS01", "Short",
     "Hello, how are you?"),

    ("TTS02", "Normal",
     "Good morning. I hope you are having a wonderful day."),

    ("TTS03", "Numbers",
     "The meeting starts at nine thirty and finishes at eleven forty five."),

    ("TTS04", "Technical",
     "Artificial intelligence can convert written text into natural sounding speech."),

    ("TTS05", "Technical",
     "Voice activity detection identifies when a person starts and stops speaking."),

    ("TTS06", "Project",
     "This local system combines speech recognition and text to speech technology."),

    ("TTS07", "Long",
     "The purpose of this experiment is to evaluate the latency efficiency and resource consumption of a completely local text to speech system."),

    ("TTS08", "Long",
     "A virtual human should be able to listen to a user understand spoken language generate a response and produce understandable speech.")
]


print("=" * 65)
print("PERSISTENT PIPER TTS BENCHMARK")
print("=" * 65)

# ------------------------------------------------------------
# Load Piper ONCE
# ------------------------------------------------------------

print("\nLoading Piper voice...")

load_start = time.perf_counter()

voice = PiperVoice.load(str(VOICE_PATH))

load_time = time.perf_counter() - load_start

print(f"Voice loaded in {load_time:.3f} seconds.\n")


results = []

# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------

for test_id, category, text in TESTS:

    output_file = f"persistent_{test_id}.wav"

    print("-" * 65)
    print(f"{test_id} - {category}")
    print(text)

    start = time.perf_counter()

    with wave.open(output_file, "wb") as wav_file:
        voice.synthesize_wav(
            text,
            wav_file
        )

    synthesis_time = time.perf_counter() - start

    # Measure generated audio duration
    with wave.open(output_file, "rb") as wav_file:

        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()

        audio_duration = (
            frames / float(sample_rate)
        )

    rtf = (
        synthesis_time / audio_duration
        if audio_duration > 0
        else 0
    )

    word_count = len(text.split())

    print(
        f"Synthesis time: {synthesis_time:.3f} s"
    )

    print(
        f"Audio duration:  {audio_duration:.3f} s"
    )

    print(
        f"TTS RTF:         {rtf:.3f}"
    )

    results.append({
        "test_id": test_id,
        "category": category,
        "text": text,
        "word_count": word_count,
        "synthesis_time_seconds": round(
            synthesis_time, 3
        ),
        "audio_duration_seconds": round(
            audio_duration, 3
        ),
        "tts_rtf": round(
            rtf, 3
        )
    })


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    fieldnames = [
        "test_id",
        "category",
        "text",
        "word_count",
        "synthesis_time_seconds",
        "audio_duration_seconds",
        "tts_rtf"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

avg_synthesis = np.mean([
    r["synthesis_time_seconds"]
    for r in results
])

avg_audio = np.mean([
    r["audio_duration_seconds"]
    for r in results
])

avg_rtf = np.mean([
    r["tts_rtf"]
    for r in results
])


print("\n")
print("=" * 65)
print("PERSISTENT TTS SUMMARY")
print("=" * 65)

print(
    f"Initial model load:      "
    f"{load_time:.3f} s"
)

print(
    f"Average synthesis time: "
    f"{avg_synthesis:.3f} s"
)

print(
    f"Average audio duration: "
    f"{avg_audio:.3f} s"
)

print(
    f"Average TTS RTF:        "
    f"{avg_rtf:.3f}"
)

print("=" * 65)

print(
    f"\nResults saved to: {OUTPUT_CSV}"
)
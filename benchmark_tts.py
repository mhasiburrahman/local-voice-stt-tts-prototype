import subprocess
import sys
import time
import wave
import os
import csv
import psutil
import numpy as np

VOICE = "en_GB-alan-medium"
VOICE_DIR = "voices"

TESTS = [
    {
        "id": "TTS01",
        "category": "Short",
        "text": "Hello, how are you?"
    },
    {
        "id": "TTS02",
        "category": "Normal",
        "text": "Good morning. I hope you are having a wonderful day."
    },
    {
        "id": "TTS03",
        "category": "Numbers",
        "text": "The meeting starts at nine thirty and finishes at eleven forty five."
    },
    {
        "id": "TTS04",
        "category": "Technical",
        "text": "Artificial intelligence can convert written text into natural sounding speech."
    },
    {
        "id": "TTS05",
        "category": "Technical",
        "text": "Voice activity detection identifies when a person starts and stops speaking."
    },
    {
        "id": "TTS06",
        "category": "Project",
        "text": "This local system combines speech recognition and text to speech technology."
    },
    {
        "id": "TTS07",
        "category": "Long",
        "text": "The purpose of this experiment is to evaluate the latency efficiency and resource consumption of a completely local text to speech system."
    },
    {
        "id": "TTS08",
        "category": "Long",
        "text": "A virtual human should be able to listen to a user understand spoken language generate a response and produce understandable speech."
    }
]

OUTPUT_CSV = "tts_benchmark_results.csv"

process = psutil.Process(os.getpid())

results = []

print("=" * 65)
print("PIPER TTS BENCHMARK")
print("=" * 65)

for test in TESTS:

    output_file = f"{test['id']}.wav"

    print("\n" + "-" * 65)
    print(f"{test['id']} - {test['category']}")
    print(test["text"])

    ram_before = process.memory_info().rss / (1024 ** 2)

    start = time.perf_counter()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper",
            "-m",
            VOICE,
            "--data-dir",
            VOICE_DIR,
            "-f",
            output_file,
            "--",
            test["text"]
        ],
        check=True,
        stdout=subprocess.DEVNULL
    )

    synthesis_time = time.perf_counter() - start

    ram_after = process.memory_info().rss / (1024 ** 2)

    # Read duration of generated WAV
    with wave.open(output_file, "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        audio_duration = frames / float(sample_rate)

    # TTS real-time factor
    tts_rtf = (
        synthesis_time / audio_duration
        if audio_duration > 0
        else 0
    )

    words = len(test["text"].split())

    print(f"Synthesis time: {synthesis_time:.3f} s")
    print(f"Audio duration:  {audio_duration:.3f} s")
    print(f"TTS RTF:         {tts_rtf:.3f}")
    print(f"Word count:      {words}")

    results.append({
        "test_id": test["id"],
        "category": test["category"],
        "text": test["text"],
        "word_count": words,
        "synthesis_time_seconds": round(synthesis_time, 3),
        "audio_duration_seconds": round(audio_duration, 3),
        "tts_rtf": round(tts_rtf, 3),
        "process_ram_mb": round(ram_after, 1),
        "ram_change_mb": round(ram_after - ram_before, 1)
    })


# Save CSV

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
        "tts_rtf",
        "process_ram_mb",
        "ram_change_mb"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# Summary

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

avg_ram = np.mean([
    r["process_ram_mb"]
    for r in results
])


print("\n")
print("=" * 65)
print("TTS BENCHMARK SUMMARY")
print("=" * 65)

print(f"Voice:                  {VOICE}")
print(f"Tests:                  {len(results)}")
print(f"Average synthesis time: {avg_synthesis:.3f} s")
print(f"Average audio duration: {avg_audio:.3f} s")
print(f"Average TTS RTF:        {avg_rtf:.3f}")
print(f"Average process RAM:    {avg_ram:.1f} MB")

print("=" * 65)

print(
    f"\nDetailed results saved to: {OUTPUT_CSV}"
)
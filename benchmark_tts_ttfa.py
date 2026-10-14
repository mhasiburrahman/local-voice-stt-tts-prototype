import time
import csv
import numpy as np
from pathlib import Path
from piper import PiperVoice

VOICE_PATH = Path("voices/en_GB-alan-medium.onnx")
OUTPUT_CSV = "tts_ttfa_results.csv"

TESTS = [
    ("TTFA01", "Short",
     "Hello, how are you?"),

    ("TTFA02", "Normal",
     "Good morning. I hope you are having a wonderful day."),

    ("TTFA03", "Numbers",
     "The meeting starts at nine thirty and finishes at eleven forty five."),

    ("TTFA04", "Technical",
     "Artificial intelligence can convert written text into natural sounding speech."),

    ("TTFA05", "Technical",
     "Voice activity detection identifies when a person starts and stops speaking."),

    ("TTFA06", "Project",
     "This local system combines speech recognition and text to speech technology."),

    ("TTFA07", "Long",
     "The purpose of this experiment is to evaluate the latency efficiency and resource consumption of a completely local text to speech system."),

    ("TTFA08", "Long",
     "A virtual human should be able to listen to a user understand spoken language generate a response and produce understandable speech.")
]

print("=" * 65)
print("PIPER TIME-TO-FIRST-AUDIO BENCHMARK")
print("=" * 65)

print("\nLoading Piper voice...")

load_start = time.perf_counter()

voice = PiperVoice.load(str(VOICE_PATH))

load_time = time.perf_counter() - load_start

print(f"Voice loaded in {load_time:.3f} seconds.")

# Warm-up
print("Running warm-up synthesis...")

for _ in voice.synthesize("Hello."):
    pass

print("Warm-up complete.\n")

results = []

for test_id, category, text in TESTS:

    print("-" * 65)
    print(f"{test_id} - {category}")
    print(text)

    start = time.perf_counter()

    first_chunk_time = None
    chunk_count = 0

    for audio_chunk in voice.synthesize(text):

        now = time.perf_counter()

        if first_chunk_time is None:
            first_chunk_time = now

        chunk_count += 1

    finish = time.perf_counter()

    ttfa = first_chunk_time - start
    total_time = finish - start

    print(f"TTFA:              {ttfa:.3f} s")
    print(f"Total generation:  {total_time:.3f} s")
    print(f"Audio chunks:      {chunk_count}")

    results.append({
        "test_id": test_id,
        "category": category,
        "text": text,
        "ttfa_seconds": round(ttfa, 3),
        "total_generation_seconds": round(total_time, 3),
        "audio_chunks": chunk_count
    })


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
        "ttfa_seconds",
        "total_generation_seconds",
        "audio_chunks"
    ]

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


avg_ttfa = np.mean([
    result["ttfa_seconds"]
    for result in results
])

avg_total = np.mean([
    result["total_generation_seconds"]
    for result in results
])

print("\n")
print("=" * 65)
print("TTFA BENCHMARK SUMMARY")
print("=" * 65)

print(f"Initial model load:       {load_time:.3f} s")
print(f"Average TTFA:             {avg_ttfa:.3f} s")
print(f"Average total generation: {avg_total:.3f} s")

print("=" * 65)

print(f"\nResults saved to: {OUTPUT_CSV}")
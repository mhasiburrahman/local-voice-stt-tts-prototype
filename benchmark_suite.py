import sounddevice as sd
import numpy as np
import time
import csv
import os
import psutil
from faster_whisper import WhisperModel
from jiwer import wer

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
MIC_DEVICE = 1

MODELS = ["tiny.en", "base.en"]

TESTS = [
    {
        "id": "T01",
        "category": "Short / Normal",
        "text": "Hello, how are you today?"
    },
    {
        "id": "T02",
        "category": "Normal",
        "text": "The quick brown fox jumps over the lazy dog."
    },
    {
        "id": "T03",
        "category": "Numbers",
        "text": "The meeting starts at nine thirty and finishes at eleven forty five."
    },
    {
        "id": "T04",
        "category": "Technical",
        "text": "Artificial intelligence can convert human speech into written text."
    },
    {
        "id": "T05",
        "category": "Technical",
        "text": "Voice activity detection identifies when a person starts and stops speaking."
    },
    {
        "id": "T06",
        "category": "Project terminology",
        "text": "The local system uses speech recognition and text to speech technology."
    },
    {
        "id": "T07",
        "category": "Long sentence",
        "text": "The purpose of this experiment is to evaluate the accuracy latency and efficiency of a local speech recognition system."
    },
    {
        "id": "T08",
        "category": "Names / Places",
        "text": "Hasibur travelled from London to Cardiff on Saturday morning."
    },
    {
        "id": "T09",
        "category": "Fast speech",
        "text": "I need to finish my university project before the meeting tomorrow."
    },
    {
        "id": "T10",
        "category": "Slow speech",
        "text": "This sentence is being spoken slowly and clearly."
    }
]

OUTPUT_CSV = "stt_benchmark_results.csv"

# ============================================================
# AUDIO RECORDING
# ============================================================

def record_test(test):

    print("\n" + "=" * 70)
    print(f"{test['id']} - {test['category']}")
    print("=" * 70)

    print("\nREAD THIS EXACTLY:")
    print(test["text"])

    input("\nPress ENTER when ready...")

    print("\nRecording...")
    print("Speak the sentence and then press ENTER.")

    chunks = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE,
        callback=callback
    ):
        input()

    audio = np.concatenate(chunks).flatten()

    # Basic silence trimming
    threshold = 0.01

    active = np.where(
        np.abs(audio) > threshold
    )[0]

    if len(active) > 0:
        audio = audio[
            active[0]:active[-1] + 1
        ]

    duration = len(audio) / SAMPLE_RATE

    print(
        f"Recorded duration: {duration:.2f} seconds"
    )

    return audio, duration


# ============================================================
# RECORD ALL TESTS ONCE
# ============================================================

print("=" * 70)
print("LOCAL STT MULTI-UTTERANCE BENCHMARK")
print("=" * 70)

print("\nYou will record 10 sentences.")
print("Each recording will be tested against BOTH models.")
print("\nFor T09, deliberately speak faster than normal.")
print("For T10, deliberately speak slowly and clearly.")

recordings = []

for test in TESTS:

    audio, duration = record_test(test)

    recordings.append({
        "test": test,
        "audio": audio,
        "duration": duration
    })


# ============================================================
# RUN MODELS
# ============================================================

results = []

process = psutil.Process(os.getpid())

for model_name in MODELS:

    print("\n")
    print("#" * 70)
    print(f"LOADING MODEL: {model_name}")
    print("#" * 70)

    load_start = time.perf_counter()

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8"
    )

    load_time = time.perf_counter() - load_start

    print(
        f"Model loaded in {load_time:.2f} seconds."
    )

    for item in recordings:

        test = item["test"]
        audio = item["audio"]
        duration = item["duration"]

        print(
            f"\nTesting {model_name} - {test['id']}..."
        )

        ram_before = (
            process.memory_info().rss
            / (1024 ** 2)
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

        latency = time.perf_counter() - start

        ram_after = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        error = wer(
            test["text"].lower(),
            transcription.lower()
        ) * 100

        rtf = (
            latency / duration
            if duration > 0
            else 0
        )

        print(f"Reference:     {test['text']}")
        print(f"Transcription: {transcription}")

        print(
            f"WER: {error:.2f}% | "
            f"Latency: {latency:.2f}s | "
            f"RTF: {rtf:.3f}"
        )

        results.append({
            "test_id": test["id"],
            "category": test["category"],
            "model": model_name,
            "reference": test["text"],
            "transcription": transcription,
            "audio_duration_seconds": round(
                duration, 3
            ),
            "stt_latency_seconds": round(
                latency, 3
            ),
            "rtf": round(
                rtf, 3
            ),
            "wer_percent": round(
                error, 2
            ),
            "process_ram_mb": round(
                ram_after, 1
            ),
            "ram_change_mb": round(
                ram_after - ram_before, 1
            )
        })

    del model


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
        "test_id",
        "category",
        "model",
        "reference",
        "transcription",
        "audio_duration_seconds",
        "stt_latency_seconds",
        "rtf",
        "wer_percent",
        "process_ram_mb",
        "ram_change_mb"
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

print("\n")
print("=" * 75)
print("BENCHMARK SUMMARY")
print("=" * 75)

for model_name in MODELS:

    model_results = [
        r for r in results
        if r["model"] == model_name
    ]

    avg_wer = np.mean([
        r["wer_percent"]
        for r in model_results
    ])

    avg_latency = np.mean([
        r["stt_latency_seconds"]
        for r in model_results
    ])

    avg_rtf = np.mean([
        r["rtf"]
        for r in model_results
    ])

    avg_ram = np.mean([
        r["process_ram_mb"]
        for r in model_results
    ])

    print(f"\n{model_name}")

    print(
        f"Average WER:     "
        f"{avg_wer:.2f}%"
    )

    print(
        f"Average latency: "
        f"{avg_latency:.3f} s"
    )

    print(
        f"Average RTF:     "
        f"{avg_rtf:.3f}"
    )

    print(
        f"Average process RAM: "
        f"{avg_ram:.1f} MB"
    )


print("\n" + "=" * 75)

print(
    f"Detailed results saved to: "
    f"{OUTPUT_CSV}"
)

print("=" * 75)
import torch
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

# === CẤU HÌNH LOCAL MODEL ===
cache_dir = "direction/of/PhoWhisper"
model_id = "vinai/PhoWhisper-tiny"

# === LOAD TỪ LOCAL CACHE ===
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)

asr = AutomaticSpeechRecognitionPipeline(model=model, processor=processor)

# === AUDIO CONFIG ===
samplerate = 16000
channels = 1
record_duration = 5
recognized_texts = []

print(" Xin chào! Tôi có thể giúp gì cho bạn?...")

def record_and_transcribe():
    print("\n🎙️ Ghi âm...")
    recording = sd.rec(
        int(record_duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype='float32'
    )
    sd.wait()

    # Ép thành 1D array (nếu bị shape (N, 1))
    audio = np.squeeze(recording)

    result = asr(audio, generate_kwargs={"language": "vi"})
    text = result["text"]
    print("📄 Bạn nói: ", text)
    recognized_texts.append(text)

try:
    while True:
        record_and_transcribe()

except KeyboardInterrupt:
    print("\n🛑 Đã dừng chương trình.")
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(recognized_texts))
    print("✅ Đã lưu kết quả vào: output.txt")

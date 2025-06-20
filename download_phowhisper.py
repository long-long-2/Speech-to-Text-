from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

target_dir = "/home/long/speech_to_text/PhoWhisper"

# Tải model và processor về thư mục đích
AutoModelForSpeechSeq2Seq.from_pretrained("vinai/PhoWhisper-tiny", cache_dir=target_dir)
AutoProcessor.from_pretrained("vinai/PhoWhisper-tiny", cache_dir=target_dir)

print("✅ Đã tải mô hình và lưu tại:", target_dir)

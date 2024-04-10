
import ssl
import torch
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import whisperx
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context
SAMPLING_RATE = 16000
WAV_FILE = 'en.wav'
device = "cuda" 
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
USE_ONNX = False
def sub_audio(audio, start, stop):
    f1 = int(start)
    f2 = int(stop)
    return audio[f1:f2]

######### 0 Preload Models ######
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=USE_ONNX)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
spk_embd_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
whisper_model = whisperx.load_model("medium", device, compute_type=compute_type)

############# 1 VAD #############
wav = read_audio(WAV_FILE, sampling_rate=SAMPLING_RATE)
speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLING_RATE)

############# 2 SPEAKER EMBEDDING ###########
audios = []
total_audio = wav.numpy()
print(type(total_audio[0]))
if not isinstance(total_audio.flat[0], np.floating):
    total_audio = total_audio.astype(np.float32) / 32768.0
for seg in speech_timestamps:
    audio = sub_audio(total_audio, seg['start'], seg['end'])
    audios.append(audio)
inputs = feature_extractor(audios, padding=True, return_tensors="pt")
embeddings = spk_embd_model(**inputs).embeddings
embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
speaker_change_flags = []
cosine_sim = torch.nn.CosineSimilarity(dim=-1)
threshold = 0.5  # the optimal threshold is dataset-dependent
for i in range(len(embeddings)):
    if i == 0:
        speaker_change_flags.append(False)
        continue
    similarity = cosine_sim(embeddings[i-1], embeddings[i])
    if similarity < threshold:
        speaker_change_flags.append(True)
    else:
        speaker_change_flags.append(False)

############# 3 ASR ##########
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
vad_segments = []
vad_segment = {}
speech_timestamps_secs = []
for speech_timestamp in speech_timestamps:
    speech_timestamp_secs = speech_timestamp
    speech_timestamp_secs['start'] = speech_timestamp_secs['start'] / SAMPLING_RATE
    speech_timestamp_secs['end'] = speech_timestamp_secs['end'] / SAMPLING_RATE
    speech_timestamps_secs.append(speech_timestamp_secs)
vad_segment['start'] = speech_timestamps_secs[0]['start']
vad_segment['end'] = speech_timestamps_secs[0]['end']
vad_segment['segments'] = [
    (speech_timestamps_secs[0]['start'], speech_timestamps_secs[0]['end'])
    ]
continuous_threshold = 2 # 2 seconds
for i in range(len(speaker_change_flags)):
    if i == 0:
        continue
    if i == len(speaker_change_flags) - 1:
        vad_segment['end'] = speech_timestamps_secs[i]['end']
        vad_segment['segments'].append((speech_timestamps_secs[i]['start'],
                                        speech_timestamps_secs[i]['end']))
        vad_segments.append(vad_segment.copy())
        continue
    if speaker_change_flags[i] or \
       (speech_timestamps_secs[i]['start'] - speech_timestamps_secs[i - 1]['end']) > continuous_threshold:
        vad_segments.append(vad_segment.copy())
        vad_segment = {}
        vad_segment['segments'] = []
        vad_segment['start'] = speech_timestamps_secs[i]['start']
        vad_segment["end"] = speech_timestamps_secs[i]['end']
        vad_segment['segments'].append((speech_timestamps_secs[i]['start'],
                                        speech_timestamps_secs[i]['end']))
    else:
        vad_segment["end"] = speech_timestamps_secs[i]['end']
        vad_segment['segments'].append((speech_timestamps_secs[i]['start'],
                                        speech_timestamps_secs[i]['end']))
print("vad_segment:", vad_segment)
result = whisper_model.transcribe(total_audio, batch_size=batch_size, vad_segments=vad_segments)
print(result["segments"]) # before alignment

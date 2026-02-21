"Script to transcribe files on source directory and write transcripts into target directory."
import os
import sys
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import dotenv

# Setup

print("THIS WORKS????")

loaded_dotenv = dotenv.load_dotenv()
pind = 4  # printing indentation

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise EnvironmentError(
        "Store the huggingface token in the .env file with key HF_TOKEN.")

print(f"{'':<{pind}}Loading pyannote...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                    token=HF_TOKEN)
print(f"{'':<{pind}}pyannote loaded")

assert pipeline is not None, "Something happened"

if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

print(f"{'':<{pind}}Loading whisper...")
whisper_model = whisper.load_model("tiny")
print(f"{'':<{pind}}Whisper loaded")

source_dir = sys.argv[1]
target_dir = sys.argv[2]

print(sys.argv)

print(source_dir, target_dir)

print("HERE TOO?")

print(os.listdir(source_dir))

for filename in os.listdir(source_dir):
    # only transcribe audio files

    file_ending = filename.split(".")[-1]
    if file_ending not in ["m4a", "mp3", "wav"]:
        print(f"{'':<{pind}}Skipping {filename}")
        continue

    print(f"{'':<{pind}}Processing {filename}")
    pind += 1

    file_path = os.path.join(source_dir, filename)

    # Speaker diarization

    print(f"{'':<{pind}}Starting speaker diarization...")

    with ProgressHook() as hook:
        waveform, sample_rate = torchaudio.load(file_path)
        speaker_diarization = pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate
            }, hook=hook).speaker_diarization

    print(f"{'':<{pind}}Speaker diarization complete")

    # Transcription

    print(f"{'':<{pind}}Starting audio transcription...")

    result = whisper_model.transcribe(file_path)

    print(f"{'':<{pind}}Transcription completed")

    # Matching diarization and transcription

    print(f"{'':<{pind}}Matching diarization and transcription...")

    transcription = ""

    whisper_index = 0

    for turn, speaker in speaker_diarization:
        transcription += f"[{speaker}]\n"
        found_end = False
        while not found_end:
            current_segment = result["segments"][whisper_index]

            w_start = current_segment["start"]
            w_end = current_segment["end"]

            p_end = turn.end

            if w_end < p_end:
                transcription += f"{current_segment['text']} "
                whisper_index += 1

            elif w_end - p_end < p_end - w_start:
                transcription += f"{current_segment['text']}\n\n"
                whisper_index += 1
                found_end = True

            else:
                transcription += "\n"
                found_end = True

    target_filename = ''.join(filename.split('.')[:-1]) + ".txt"

    target_file_path = os.path.join(target_dir, target_filename)

    with open(f"{target_file_path}", "w",
              encoding="utf-8") as file_txt:
        file_txt.write(transcription)

    print(
        f"{'':<{pind}}Diarization and transcription matched and saved to {target_file_path}"
    )

    pind -= 1

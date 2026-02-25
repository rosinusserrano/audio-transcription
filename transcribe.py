"Script to transcribe files on source directory and write transcripts into target directory."
import os
import sys
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import dotenv

# Usage info


def print_usage_and_exit():
    print("Usage:")
    print("  transcribe.py SOURCE [TARGET]")
    print(
        "\nGoes through the source directory recursively and transcribes all audio files.\n"
    )
    print("- SOURCE: source directory where audio files are located.")
    print("- TARGET (optional): if specified creates the transcription files")
    print(
        "                     in TARGET, mirroring the folder structure in SOURCE."
    )
    exit()


source_dir = None
target_dir = None

if len(sys.argv) == 1 or len({"-h", "--help"}.intersection(sys.argv)) != 0:
    print_usage_and_exit()

elif len(sys.argv) == 2:
    source_dir = sys.argv[1]
    target_dir = source_dir
    print(f"Only source directory specified: {source_dir}.")
    print("  Writing transcriptions into same folder as audio files.")

elif len(sys.argv) == 3:
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    print(f"Got source directory: {source_dir}")
    print(f"  and target directory: {target_dir}")

else:
    print_usage_and_exit()

assert isinstance(source_dir, str), "source directory not set"
assert isinstance(target_dir, str), "target directory not set"

# Setup

loaded_dotenv = dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise EnvironmentError(
        "Store the huggingface token in the .env file with key HF_TOKEN.")

print("Loading pyannote...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                    token=HF_TOKEN)
assert pipeline is not None, "Pipeline is None"
print("pyannote loaded")

if torch.cuda.is_available():
    print("Using CUDA")
    pipeline.to(torch.device("cuda"))

print("Loading whisper...")
whisper_model = whisper.load_model("large-v3")
print("Whisper loaded")


def transcribe_directory(src: str, dest: str, depth: int):
    for filename in os.listdir(src):

        filepath = os.path.join(src, filename)

        # recursive call

        if os.path.isdir(filepath):
            target_path = os.path.join(dest, filename)
            os.makedirs(target_path, exist_ok=True)
            print(f"{'':<{depth}}Going into folder {filepath}")
            transcribe_directory(
                filepath,
                target_path,
                depth + 1,
            )
            continue

        # only transcribe audio files

        file_ending = filename.split(".")[-1]
        if file_ending not in ["m4a", "mp3", "wav"]:
            print(f"{'':<{depth}}Skipping {filepath}")
            continue

        print(f"{'':<{depth}}Processing {filepath}")

        # Speaker diarization

        print(f"{'':<{depth + 1}}Starting speaker diarization...")

        with ProgressHook() as hook:
            assert pipeline is not None, "Pipeline is None"
            waveform, sample_rate = torchaudio.load(filepath)
            speaker_diarization = pipeline(
                {
                    "waveform": waveform,
                    "sample_rate": sample_rate
                }, hook=hook).speaker_diarization

        print(f"{'':<{depth + 1}}Speaker diarization complete")

        # Transcription

        print(f"{'':<{depth + 1}}Starting audio transcription...")

        result = whisper_model.transcribe(filepath)

        print(f"{'':<{depth + 1}}Transcription completed")

        # Matching diarization and transcription

        print(f"{'':<{depth + 1}}Matching diarization and transcription...")

        transcription = ""

        whisper_index = 0

        for turn, speaker in speaker_diarization:
            speaker_header_added = False
            found_end = False
            while not found_end:
                current_segment = result["segments"][whisper_index]

                w_start = current_segment["start"]
                w_end = current_segment["end"]

                p_end = turn.end

                if w_end < p_end:
                    if not speaker_header_added:
                        transcription += f"[{speaker}] {turn}\n"
                        speaker_header_added = True
                    transcription += f"{current_segment['text']} "
                    whisper_index += 1
                    if whisper_index >= len(result["segments"]):
                        found_end = True

                elif w_end - p_end < p_end - w_start:
                    if not speaker_header_added:
                        transcription += f"[{speaker}] {turn}\n"
                        speaker_header_added = True
                    transcription += f"{current_segment['text']}"
                    whisper_index += 1
                    found_end = True

                else:
                    transcription += "\n"
                    found_end = True
            
            if speaker_header_added:
                transcription += "\n\n"

            if whisper_index >= len(result["segments"]):
                break

        target_filename = ''.join(filename.split('.')[:-1]) + ".txt"

        target_file_path = os.path.join(dest, target_filename)

        with open(f"{target_file_path}", "w", encoding="utf-8") as file_txt:
            file_txt.write(transcription)

        print(
            f"{'':<{depth + 1}}Diarization and transcription matched and saved to {target_file_path}"
        )

if __name__ == "__main__":
    transcribe_directory(source_dir, target_dir, depth=0)

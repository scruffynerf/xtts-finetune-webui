import os
# import gc
import torch
import torchaudio
import pandas as pd
from pathlib import Path
import glob
import json
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence

# from faster_whisper import WhisperModel
# import difflib

from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
tokenizer = VoiceBpeTokenizer()
char_limits = tokenizer.char_limits

# Add support for JA train
# from utils.tokenizer import multilingual_cleaners

# torch.set_num_threads(1)
torch.set_num_threads(16)

audio_types = (".wav", ".mp3", ".flac", ".opus")  # could now be anything ffmpeg supports via pydub


def find_latest_best_model(folder_path):
    search_path = Path(folder_path) / '**' / 'best_model.pth'
    files = glob.glob(str(search_path), recursive=True)
    return max(files, key=Path().stat().st_ctime_ns, default=None)


def list_audios(basePath, contains=None, validExts=audio_types):
    # loop over the directory structure
    for rootDir, dirNames, filenames in os.walk(basePath):
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = Path(filename).suffix

            # check to see if the file is an audio and should be processed
            if validExts is None or ext.lower().endswith(tuple(validExts)):
                yield Path(rootDir) / filename


def format_audio_list(audio_files, asr_model, target_language="en", out_path=None, buffer=0.2, eval_percentage=0.15, speaker_name="coqui", gradio_progress=None):
    # Initialize
    audio_total_size = 0
    if out_path is None:
        print("where are files being stored?")
    else:
        out_path = Path(out_path)
        wav_output_path = Path(out_path) / "wavs"
        wav_output_path.mkdir(parents=True, exist_ok=True)

    # Metadata setup
    metadata = {"audio_file": [], "text": [], "speaker_name": []}
    target_sr = 24000
    duration_limit = 15000

    current_language = check_language(out_path, target_language)

    # Progress tracking
    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    # Process audio files
    for audio_path in tqdm_object:
        print(f"Processing {audio_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Input file '{audio_path}' not found.")
        if existing := find_matching_files(audio_path, wav_output_path):
            print(f"... has {len(existing)} existing saved chunks, so not reprocessing")
            continue

        item_duration, metadata, current_language = process_audio(audio_path, 0, duration_limit, metadata, current_language, target_sr, asr_model, speaker_name, out_path)
        audio_total_size += item_duration

    # loop done, now handle metadata and return results
    train_metadata_path, eval_metadata_path = process_metadata(out_path, metadata, eval_percentage)
    return train_metadata_path, eval_metadata_path, audio_total_size


def process_audio(audio_path, audio_item_length, duration_limit, metadata, current_language, target_sr, asr_model, speaker_name, out_path):
    total_duration = 0
    wav_output_path = Path(out_path) / "wavs"

    audio_file_name_no_ext = Path(audio_path).stem

    audio = AudioSegment.from_file(audio_path)
    audio_chunks = split_audio(audio, duration_limit)
    num_digits = len(str(len(audio_chunks)))

    for i, chunk in enumerate(audio_chunks):
        duration_ms = len(chunk)
        audio_start = audio_item_length
        audio_end = audio_item_length + duration_ms
        duration = duration_ms // 1000
        if duration_ms > duration_limit:
            # this is after it tries to split smaller
            rejection = {
                "status": "rejected",
                "audio": str(audio_path),
                "reason": f"Duration: unable to split section from {audio_start // 1000} to {audio_end // 1000} into less than {duration}s chunk"
            }
            print(f"{rejection['status']}: {rejection['reason']}")
            write_json_results(out_path, "rejected.json", [rejection])
            continue

        wav_file_name = Path(wav_output_path) / f"{audio_file_name_no_ext}_{i:0{num_digits}d}@{duration}s_chunk.wav"
        if wav_file_name.exists():
            print(f"{wav_file_name} already exists; skipping...")
            continue
        # print(f"{audio_file_name_no_ext} chunk {i} length = {duration_ms / 1000}")

        # then we make a tensor out of the wav data in the chunk, and make it mono and 24khz
        wav_tensor, sr = audio_chunk_to_tensor(chunk, target_sr)
        # wav_seconds = wav_tensor.size(-1) / sr
        # print(f"{audio_file_name_no_ext} wav {i} length = {wav_seconds}")

        # Save the audio
        torchaudio.save(str(wav_file_name), wav_tensor.unsqueeze(0), sr)
        # print(f"saved: {wav_file_name}")

        # now we transcribe it
        # this was working with tensor passed in, I thought, but now using file
        transcribed_text, text_language = get_transcription(asr_model, wav_file_name, current_language)
        if text_language != current_language:
            print(f"Warning: Language change: {text_language}")
            current_language = text_language

        full_text, error = transcription_clean_text(
            transcribed_text,
            current_language
        )

        if full_text:
            print(f"chunk{i:0{num_digits}d}: {full_text}")

        if error:
            if full_text:
                correction = error[0]
                print(f"Corrected from: {correction['text']}")
                correction['audio'] = str(wav_file_name)
                write_json_results(out_path, "corrected.json", [correction])
            else:
                rejection = error[0]
                rejection['reason'] += f" from {audio_start} to {audio_end}"
                print(f"{rejection['status']}: {rejection['reason']}\nretrying....")
                # try to re-split it, and don't save this one into metadata (done inside loop), but do record the duration.
                duration_ms, metadata, current_language = process_audio(wav_file_name, audio_item_length, duration_limit*.75, metadata, current_language, target_sr, asr_model, speaker_name, out_path)
                wav_file_name.rename(wav_file_name.with_suffix('.resplit'))
                rejection['audio'] = str(wav_file_name)
                write_json_results(out_path, "rejected.json", [rejection])
                audio_item_length += duration_ms
                total_duration += duration_ms
                continue

        audio_item_length += duration_ms
        total_duration += duration_ms
        metadata["audio_file"].append(str(wav_file_name))
        metadata["text"].append(full_text.strip())
        metadata["speaker_name"].append(speaker_name)

    return total_duration, metadata, current_language


def audio_chunk_to_tensor(chunk, target_sr):
    # Export the chunk to a raw audio format in memory
    chunk_audio = chunk.export(format="wav")  # Export chunk to wav-like data
    # Load the chunk using torchaudio
    wav_tensor, sr = torchaudio.load(chunk_audio)
    # Convert to mono if necessary
    if wav_tensor.size(0) != 1:
        wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)
    # Resample to 24000 Hz if the sample rate is not already 24000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav_tensor = resampler(wav_tensor)
        sr = target_sr  # Update the sample rate
    # Remove channel dimension
    wav_tensor = wav_tensor.squeeze()
    # Calculate the wav duration in seconds as a double check
    return wav_tensor, sr


def find_matching_files(audio_filepath, wav_output_path):
    # Create a Path object for the audio file
    audio_file = Path(audio_filepath)

    # Extract the file name without extension
    audio_file_stem = audio_file.stem

    # Define the search pattern
    search_pattern = f"{audio_file_stem}_*_chunk.wav"

    return list(wav_output_path.glob(search_pattern))


def get_transcription(asr_model, wav_tensor, language):
    # Transcribe with Whisper
    segments, info = asr_model.transcribe(
        wav_tensor,
        vad_filter=False,
        word_timestamps=False,
        language=language
        )

    if language != info.language:
        language = info.language

    full_text = "".join(segment.text for segment in segments)
    return full_text, language


def split_audio(audio, max_chunk_length_ms=15000, min_silence_len=400, silence_thresh=-40):
    """
    Splits an audio file into chunks based on silence and merges shorter chunks.

    Parameters:
        audio (AudioSegment): Input audio to split.
        max_chunk_length_ms (int): Maximum length of a chunk in milliseconds (default: 15000ms).
        min_silence_len (int): Minimum silence duration to split audio (default: 400ms).
        silence_thresh (int): Silence threshold in dBFS (default: -40).

    Returns:
        list: A list of newly created audio chunks.
    """

    if min_silence_len < 100:
        return False

    # Split the audio file based on silence
    # print("Splitting audio based on silence...")
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=True  # Keep all silences at the boundaries
    )

    # Merge adjacent chunks shorter than the specified length
    merged_chunks = []
    current_chunk = AudioSegment.empty()
    for chunk in chunks:
        if len(chunk) > max_chunk_length_ms:
            # Attempt to re-split the chunk with a smaller minimum silence
            print(f"Found a chunk that is too large: {len(chunk)}ms. Retrying section with smaller silence: {min_silence_len - 100}ms")
            if resplit := split_audio(
                chunk,
                max_chunk_length_ms,
                (min_silence_len - 100),
                silence_thresh,
            ):
                print(f"Split into {len(resplit)} pieces")
                for splitchunk in resplit:
                    combined = len(current_chunk) + len(splitchunk)
                    if combined <= max_chunk_length_ms:
                        current_chunk += splitchunk
                    else:
                        merged_chunks.append(current_chunk)
                    current_chunk = splitchunk
            else:
                print("Unable to split the audio chunk any smaller... will accept it for now.")
                merged_chunks.extend((current_chunk, chunk))
                current_chunk = AudioSegment.empty()
        else:
            combined = len(current_chunk) + len(chunk)
            if combined <= max_chunk_length_ms:
                current_chunk += chunk
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk
    merged_chunks.append(current_chunk)
    return merged_chunks


def check_language(out_path, target_language):
    lang_file_path = out_path / "lang.txt"
    current_language = None

    if lang_file_path.exists():
        with lang_file_path.open('r', encoding='utf-8') as existing_lang_file:
            current_language = existing_lang_file.read().strip()

    if current_language != target_language:
        with lang_file_path.open('w', encoding='utf-8') as lang_file:
            lang_file.write(target_language + '\n')
        print("Warning, existing language does not match target language. Updated lang.txt with target language.")
    else:
        print("Existing language matches target language")

    return current_language


def transcription_clean_text(full_text, lang='en'):

    error = None
    char_limit = char_limits.get(lang, 250)

    # not done here (yet?)  but another option is to transcribe twice
    # (different options, or even different model)  and see if same result occurs

    # handle when whisper gets it wrong, with duplicate text
    dupecheck = remove_trailing_duplicate(full_text.strip())
    if full_text.strip() != dupecheck.strip():
        if len(dupecheck.strip()) <= char_limit:
            print("Flagging/fixing possible hallucinated duplicate...")
            error = [{"status": "duplicate text", "text": full_text.strip(), "correction": dupecheck.strip()}]
            return dupecheck.strip(), error
        else:
            # still over the char_limit...
            print(f"Rejecting for long text (more than {char_limit} chars breaks training)")
            error = [{"status": "rejected", "text": full_text.strip(), "reason": f"Length: {len(dupecheck.strip())}"}]
            return False, error
    elif full_text.strip().endswith(" you"):
        if len(full_text.strip()) <= char_limit+4:
            # handle whisper hallucination of 'you' at end of way too many lines
            print("Flagging/fixing possible 'you' hallucination")
            error = [{"status": "extra you at end", "text": full_text.strip(), "correction": "removed ' you' at end"}]
            return full_text[:-4].strip(), error
        else:
            # still over the char_limit...
            error = [{"status": "rejected", "text": full_text.strip(), "reason": f"Length: {len(full_text.strip())}"}]
            return False, error

    # length check... no longer than allowed for given language...
    if len(full_text.strip()) > char_limit:
        # print(f"Rejecting for long text (more than {char_limit} chars breaks training)")
        error = [{"status": "rejected", "text": full_text.strip(), "reason": f"Length: {len(full_text.strip())}"}]
        return False, error

    return full_text.strip(), None


def process_metadata(out_path, metadata, eval_percentage):
    train_metadata_path = out_path / "metadata_train.csv"
    eval_metadata_path = out_path / "metadata_eval.csv"
    existing_metadata = {'train': None, 'eval': None}

    for metadata_type in ['train', 'eval']:
        metadata_file = out_path / f"metadata_{metadata_type}.csv"
        if metadata_file.exists():
            existing_metadata[metadata_type] = pd.read_csv(str(metadata_file), sep="|")
            print(f"Existing {metadata_type} metadata found and loaded.")

    existing_train_df = existing_metadata['train'] if existing_metadata['train'] is not None and not existing_metadata['train'].empty else pd.DataFrame(columns=["audio_file", "text", "speaker_name"])
    existing_eval_df = existing_metadata['eval'] if existing_metadata['eval'] is not None and not existing_metadata['eval'].empty else pd.DataFrame(columns=["audio_file", "text", "speaker_name"])

    if len(metadata["audio_file"]) == 0:
        return train_metadata_path, eval_metadata_path

    # Format current metadata
    metadata_df = pd.DataFrame(metadata)

    # Shuffle and peel off the evals
    df_shuffled = metadata_df.sample(frac=1)
    num_val_samples = max(int(len(df_shuffled) * eval_percentage), 1)

    new_train_df = df_shuffled[num_val_samples:]
    final_training_set = pd.concat([existing_train_df, new_train_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    final_training_set.sort_values('audio_file').to_csv(str(train_metadata_path), sep='|', index=False)

    new_eval_df = df_shuffled[:num_val_samples]
    final_eval_set = pd.concat([existing_eval_df, new_eval_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    final_eval_set.sort_values('audio_file').to_csv(str(eval_metadata_path), sep='|', index=False)

    return train_metadata_path, eval_metadata_path


def write_json_results(out_path, filename, new_data):
    file_path = Path(out_path) / filename
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as json_file:
            existing_data = json.load(json_file)
            if isinstance(existing_data, list):
                existing_data.extend(new_data)
            else:
                raise TypeError(f"Existing data in {file_path} is not a list.")
    else:
        existing_data = new_data

    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(existing_data, json_file, indent=4, ensure_ascii=False)


def remove_trailing_duplicate(full_text):
    """
    Removes a duplicated fragment at the end of the given text.
    Only matches if the trailing duplicate has no intervening text.
    :param text: Input text with potential trailing duplicate.
    :return: Cleaned text without trailing duplicate.
    """
    words = full_text.split()
    n = len(words)

    # Start with the longest potential duplicate segment and work backward
    for i in range(n // 2, 0, -1):
        # Extract trailing and preceding segments of the same length
        trailing_segment = words[-i:]
        preceding_segment = words[-2 * i: -i]

        # Check for exact match
        if trailing_segment == preceding_segment:
            # Return text without the trailing duplicate
            print(f"Duplicate text found in:\n{full_text}")
            return ' '.join(words[:-i])

    # If no duplicates found, return original text
    return full_text

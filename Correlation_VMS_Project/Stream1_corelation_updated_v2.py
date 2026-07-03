import librosa
import os
import shutil
import numpy as np
import soundfile as sf
import threading
import re
from datetime import datetime

# Eagerly initialize librosa to prevent lazy-loading in threads during shutdown
# This avoids the "can't register atexit after shutdown" error
_ = librosa.get_samplerate

# Specify the directory containing the files - Windows/Linux compatible
base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
folder_path = os.path.join(base_path, 'Stream1_searchword1')

# Get the list of files in the folder
if os.path.exists(folder_path):
    file_list = os.listdir(folder_path)
else:
    file_list = []

GlobalPred = []

globalc = 0
word1 = ""  # Initialize word1
global_lock = threading.Lock()

# Event object to synchronize threads
concatenate_event = threading.Event()
local_counter = threading.local()

# Function to be executed in the sub-thread
def process_file(file_name):
    global globalc
    global global_lock
    global word1
    global local_counter
    local_counter.value = 0

    # Perform processing for each file
    print("Processing file:", file_name)

    # Split the file path using the backslash as a separator
    parts = file_name.split(os.sep)
    # Get the last part of the path, which contains the file name
    file_name1 = parts[-1]
    # Split the file name using the dash as a separator and extract the word after the dash
    word1 = file_name1.split('-')[-1].split('.')[0]

    # Set word signal to search for
    word_signal, sr_word = librosa.load(file_name)

    # Specify the directory containing the audio files
    audio_files_directory = os.path.join(base_path, 'Stream1audios')
    video_files_directory = os.path.join(base_path, 'Stream1videos')

    # Get a list of all the audio files in the directory
    audio_files = [file for file in os.listdir(audio_files_directory) if file.endswith(".wav")]

    # Sort the audio files in the desired order
    audio_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Iterate over the audio files
    for audio_file in audio_files:
        if not concatenate_event.is_set():  # Check if the event is not set
            # Construct the full path of the audio file
            audio_file_path = os.path.join(audio_files_directory, audio_file)
            # Load the audio file
            y, sr = librosa.load(audio_file_path)
            # Perform cross-correlation between audio and word signal
            corr = np.correlate(y, word_signal, mode='same')
            # Find the index of the maximum correlation
            idx_max = np.argmax(corr)
            # Get the start and end time of the word signal in seconds
            start_time = librosa.frames_to_time(idx_max, sr=sr)
            end_time = librosa.frames_to_time(idx_max + len(word_signal) - 1, sr=sr)
            cut_start = idx_max
            cut_end = idx_max + len(word_signal)
            # Extract the matched portion of the audio file
            match = y[cut_start:cut_end]
            matching_percentage = (np.max(corr) / (np.linalg.norm(match) * np.linalg.norm(word_signal))) * 100
            if matching_percentage > 100:
                matching_percentage = 100
            # Write the matched portion of the audio file to disk
            if matching_percentage > 70:
                print(f"Matching percentage: {matching_percentage:.2f}%")
                print("Matching percentage is greater than 70, so the audio file is saved.", audio_file)

                # Set the event to indicate that matching percentage > 70
                concatenate_event.set()

                # Get the current file number from the audio file name
                match = re.search(r"live_(\d+)\.wav", audio_file)
                if match:
                    current_file_number = int(match.group(1))
                    print("Current file number:", current_file_number)
                    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    timestamp_log_path = os.path.join(base_path, 'Stream1_detection', f"{word1}", "timestamps.txt")
                    os.makedirs(os.path.dirname(timestamp_log_path), exist_ok=True)
                    with open(timestamp_log_path, "a") as f:
                        f.write(f"[{detection_time}] Detected '{word1}' in live_{current_file_number}.mp4\n")
                    print(f"[{detection_time}] Detected '{word1}' in live_{current_file_number}.mp4")
                    
                    # Store the matched file index for later use in the main thread
                    with global_lock:
                        globalc = current_file_number
            else:
                print(f"{matching_percentage:.2f}%")
                print(audio_file)
                local_counter.value += 1
                print("local counter increment", local_counter.value)
                
            if local_counter.value > 7:
                file_number = int(audio_file_path.split("_")[1].split(".")[0])
                file_number_to_delete = file_number - 7
                # Create the file name of the file to be deleted
                file_to_delete = "live_{}.wav".format(file_number_to_delete)
                vfile_to_delete = "live_{}.mp4".format(file_number_to_delete)
                # Construct the full path of the file to be deleted
                file_path = os.path.join(audio_files_directory, file_to_delete)
                vfile_path = os.path.join(video_files_directory, vfile_to_delete)
                print("file path to delete", file_path)
                # Check if the file exists before deleting
                if os.path.isfile(file_path):
                    print("File '{}' has been deleted.".format(file_path))
                    print("File '{}' has been deleted.".format(vfile_path))
                    os.remove(file_path)
                    os.remove(vfile_path)
                else:
                    print("File '{}' does not exist.".format(file_to_delete))

# Function to concatenate files and reset the event
def perform_concatenation(start_index, end_index, output_filename):
    # Specify the directory containing the audio files
    audio_files_directory = os.path.join(base_path, 'Stream1audios')

    # Get the list of audio files in the directory
    audio_files = [file for file in os.listdir(audio_files_directory) if file.endswith(".wav")]
    audio_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
   
    video_files_directory = os.path.join(base_path, 'Stream1videos')

    # Get the list of video files in the directory
    video_files = [file for file in os.listdir(video_files_directory) if file.endswith(".mp4")]
    video_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
   
    while f'live_{start_index}.mp4' not in video_files:
        start_index += 1

    # Get the start and end index of the original files for concatenation
    start_concat = start_index
    end_concat = end_index
    print(start_concat)
    print(end_concat)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    with open(output_filename, 'ab') as outfile:
        for i in range(start_concat, end_concat + 1):
            print(i)
            filename = os.path.join(video_files_directory, f'live_{i}.mp4')
            if os.path.exists(filename):
                with open(filename, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)

    # Delete the concatenated files
    delete_files(start_concat, end_index, audio_files_directory, video_files_directory)

    # Reset the event to allow threads to process again
    concatenate_event.clear()

# Function to delete files
def delete_files(start_index, end_index, audio_files_directory, video_files_directory):
    import time
    
    def safe_remove(filepath, max_retries=3):
        """Try to remove a file with retries for permission errors"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Wait 500ms before retrying
                    continue
                else:
                    return False  # Give up after max retries
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")
                return False
        return False
    
    for i in range(1, start_index):
        filename = os.path.join(audio_files_directory, f'live_{i}.wav')
        vfilename = os.path.join(video_files_directory, f'live_{i}.mp4')
        safe_remove(filename)
        safe_remove(vfilename)
    
    for i in range(start_index, end_index + 1):
        filename = os.path.join(audio_files_directory, f'live_{i}.wav')
        vfilename = os.path.join(video_files_directory, f'live_{i}.mp4')
        safe_remove(filename)
        safe_remove(vfilename)

# Function to be executed in the main thread
def main_thread_function():
    global globalc
    global global_lock
    global word1

    while True:
        # Start the sub-threads
        sub_threads = []
        for file_name in file_list:
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            # Create a sub-thread for each file
            sub_thread = threading.Thread(target=process_file, args=(file_path,))
            sub_threads.append(sub_thread)
            sub_thread.start()

        # Wait for all sub-threads to finish
        for sub_thread in sub_threads:
            sub_thread.join()

        if concatenate_event.is_set():
            # Acquire the lock to modify the global counter
            global_lock.acquire()

            # Get the file index from the event
            processing_index = globalc

            # Release the lock
            global_lock.release()

            if processing_index is not None:
                # Define the range of files to concatenate (5 before and 5 after the matched file)
                start_index = max(0, processing_index - 5)
                end_index = processing_index + 5

                # Construct the output file name for concatenation
                output_filename = os.path.join(base_path, 'Stream1_detection', f'{word1}/{word1}_detected_at_{processing_index}.mp4')

                # Perform concatenation and deletion in the main thread
                perform_concatenation(start_index, end_index, output_filename)

                # Reset the event to allow sub-threads to process again
                concatenate_event.clear()
        else:
            # No matching percentage > 70 found, exit the loop
            break

# Create and start the main thread
if __name__ == "__main__" and file_list:
    main_thread = threading.Thread(target=main_thread_function)
    main_thread.start()
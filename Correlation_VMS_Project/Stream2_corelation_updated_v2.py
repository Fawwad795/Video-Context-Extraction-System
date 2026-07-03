import os
import shutil
import numpy as np
import librosa
import soundfile as sf
import threading
import re
from datetime import datetime

# Specify the directory containing the files - Windows/Linux compatible
base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
folder_path = os.path.join(base_path, 'Stream2_searchword1')

# Get the list of files in the folder
file_list = os.listdir(folder_path)

GlobalPred = []

globalc = 0
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
    ####### below line was as: parts = file_name.split('\\')
    parts = file_name.split('/')
    # Get the last part of the path, which contains the file name
    file_name1 = parts[-1]
    # Split the file name using the dash as a separator and extract the word after the dash
    word1 = file_name1.split('-')[-1].split('.')[0]

    # Set word signal to search for
    word_signal, sr_word = librosa.load(file_name)

    # Specify the directory containing the audio files
    audio_files_directory = os.path.join(base_path, 'Stream2audios')
    video_files_directory = os.path.join(base_path, 'Stream2videos')

    # Get a list of all the audio files in the directory
    audio_files = [file for file in os.listdir(audio_files_directory) if file.endswith(".wav")]

    # Sort the audio files in the desired order
    audio_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Iterate over the audio files
    for audio_file in audio_files:
        if not concatenate_event.is_set():  # Check if the event is not set (matching percentage > 50)
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
            # Write the matched portion of the audio file to disk
            if matching_percentage > 65:
                print(f"Matching percentage: {matching_percentage:.2f}%")
                print("Matching percentage is greater than 65, so the audio file is saved.", audio_file)

                # Set the event to indicate that matching percentage > 50, and stop other threads
                concatenate_event.set()

                # Get the current file number from the audio file name
                match = re.search(r"live_(\d+)\.wav", audio_file) ######################################################################
                if match:
                    current_file_number = int(match.group(1))
                    print("Current file number:", current_file_number)
                    # Store the matched file index for later use in the main thread
                    with global_lock:
                        globalc = current_file_number
                        word1 = word1

            else:
                print(f"{matching_percentage:.2f}%")
                print(audio_file)
                local_counter.value += 1
                print("local counter increment",local_counter.value)
                
                if local_counter.value > 7:
                    file_number = int(audio_file_path.split("_")[1].split(".")[0])
                    file_number_to_delete = file_number - 7
                    # Create the file name of the file to be deleted
                    file_to_delete = "live_{}.wav".format(file_number_to_delete)
                    vfile_to_delete = "live_{}.mp4".format(file_number_to_delete)
                    # Construct the full path of the file to be deleted
                    file_path = os.path.join(audio_files_directory, file_to_delete)
                    vfile_path = os.path.join(video_files_directory, vfile_to_delete)
                    print("file path to delete",file_path)
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
    audio_files_directory = r"/home/jetson/VMS/GUI2CHjetson/Stream2audios/"

    # Get the list of audio files in the directory
    audio_files = [file for file in os.listdir(audio_files_directory) if file.endswith(".wav")]
    audio_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
   
    video_files_directory = r"/home/jetson/VMS/GUI2CHjetson/Stream2videos/"

    # Get the list of audio files in the directory
    video_files = [file for file in os.listdir(video_files_directory) if file.endswith(".mp4")]
    video_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
   
    while f'live_{start_index}.mp4' not in video_files:
        start_index += 1

    # Get the start and end index of the original files for concatenation
    start_concat = start_index
    end_concat = end_index
    print(start_concat)
    print(end_concat)

   
    with open(output_filename, 'ab') as outfile:
        for i in range(start_concat, end_concat + 1):
            print(i)
            filename = os.path.join(video_files_directory, f'live_{i}.mp4')
            with open(filename, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)

    # Delete the concatenated files
    delete_files(start_concat, end_index, audio_files_directory,video_files_directory)

    # Reset the event to allow threads to process again
    concatenate_event.clear()

# Function to delete files
def delete_files(start_index, end_index, audio_files_directory, video_files_directory):
    for i in range(1, start_index):
        filename = os.path.join(audio_files_directory, f'live_{i}.wav')
        vfilename = os.path.join(video_files_directory, f'live_{i}.mp4')
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(vfilename):
            os.remove(vfilename)
    for i in range(start_index, end_index + 1):
        filename = os.path.join(audio_files_directory, f'live_{i}.wav')
        vfilename = os.path.join(video_files_directory, f'live_{i}.mp4')
        os.remove(filename)
        os.remove(vfilename)

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
            word1 = word1

            # Release the lock
            global_lock.release()

            if processing_index is not None:
                # Define the range of files to concatenate (5 before and 5 after the matched file)
                start_index = max(0, processing_index - 5)
                end_index = processing_index + 5

                output_filename = os.path.join("/home/jetson/VMS/GUI2CHjetson/Stream2_detection/", f'{word1}/{word1}_detected_at_{processing_index}.mp4')

                # Perform concatenation and deletion in the main thread
                perform_concatenation(start_index, end_index, output_filename)

                # Reset the event to allow sub-threads to process again
                concatenate_event.clear()

        else:
            # No matching percentage > 50 found, exit the loop
            break

# Create and start the main thread
main_thread = threading.Thread(target=main_thread_function)
main_thread.start()

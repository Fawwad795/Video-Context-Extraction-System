import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess
import os
import shutil
import threading
import webbrowser
from datetime import datetime
from pytube import YouTube
import time

class TextToSpeechStream:
    def __init__(self, root, stream_name, row):
        self.root = root
        self.stream_name = stream_name
        self.row = row
        self.is_running = False
        self.monitoring_active = False  # Initialize monitoring flag

        self.create_widgets()

    def create_widgets(self):
        # Create a frame for the stream
        self.stream_frame = ttk.Frame(self.root, padding="20", style="Stream.TFrame")
        self.stream_frame.grid(row=self.row, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        # Create a label for the stream name
        self.stream_label = ttk.Label(self.stream_frame, text=self.stream_name, font=("Helvetica", 16, "bold"), style="Stream.TLabel")
        self.stream_label.grid(row=0, column=0, columnspan=4, pady=10, sticky="w")

        # Create a label for instructions above Search Word
        ttk.Label(self.stream_frame, text="                                Enter word(s) or list separated by commas", style="Instruction.TLabel").grid(row=1, column=0, columnspan=2, padx=5, pady=(5,0), sticky="w")

        # Create a label and entry for user input (word)
        ttk.Label(self.stream_frame, text="Search Word:", style="Stream.TLabel").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.word_entry = ttk.Entry(self.stream_frame, width=30)
        self.word_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Create a label for instructions above URL
        ttk.Label(self.stream_frame, text="                         Input live stream Youtube URL only", style="Instruction.TLabel").grid(row=1, column=2, columnspan=2, padx=5, pady=(5,0), sticky="w")

        # Create a label and entry for user input (URL)
        ttk.Label(self.stream_frame, text="URL:", style="Stream.TLabel").grid(row=2, column=2, padx=5, pady=5, sticky="e")
        self.url_entry = ttk.Entry(self.stream_frame, width=30)
        self.url_entry.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        # Create a label to display creation time
        self.creation_time_label = ttk.Label(self.stream_frame, text="", style="Stream.TLabel")
        self.creation_time_label.grid(row=2, column=4, columnspan=2, padx=5, pady=5, sticky="w")

        # Create a label to display channel name
        self.channel_label = ttk.Label(self.stream_frame, text="", style="Stream.TLabel")
        self.channel_label.grid(row=3, column=4, columnspan=2, padx=5, pady=5, sticky="w")

        # Create a label for status
        self.status_label = ttk.Label(self.stream_frame, text="Status: ", style="Stream.TLabel")
        self.status_label.grid(row=4, column=0, columnspan=4, pady=5, sticky="w")

        # Create a button to trigger text-to-speech and URL processing
        self.action_button = ttk.Button(self.stream_frame, text="Start", command=self.toggle_processing, style="Stream.TButton")
        self.action_button.grid(row=5, column=1, pady=10, sticky="n")

        # Create a "View" button to open the folder
        self.view_button = ttk.Button(self.stream_frame, text="View", command=self.open_folder, style="Stream.TButton")
        self.view_button.grid(row=5, column=2, pady=10, sticky="n")

        # Create a label to display the count of .mp4 videos
        self.mp4_count_label = ttk.Label(self.stream_frame, text="", style="Stream.TLabel")
        self.mp4_count_label.grid(row=5, column=3, columnspan=2, padx=5, pady=5, sticky="e")

    def toggle_processing(self):
        if self.is_running:
            self.stop_processing()
            self.is_running = False
        else:
            self.start_processing()
            self.is_running = True

    def start_processing(self):
        self.action_button.config(text="Stop", style="Running.TButton")
        self.status_label.config(text="Status: Generating synthesized words, please wait...")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.creation_time_label.config(text=f"    Created on: {current_time}")
        
        # Get the word and URL from the entries
        words = self.word_entry.get().split(',')
        # Check if either word or URL is empty
        if not any(words):
            self.status_label.config(text="Status: Please enter at least one word.")
            return

        for word in words:
            if not word.strip():
                continue
            base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
            word_folder = os.path.join(base_path, f'{self.stream_name}_detection', word.strip())
            os.makedirs(word_folder, exist_ok=True)
            
            self.status_label.config(text="Status: Generating synthesized words, please wait...")

            try:
                python_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'Scripts', 'python.exe')
                subprocess.run([python_exe, f"{self.stream_name}_hifigan.py", word.strip()], check=True)
                self.status_label.config(text=f"Status: Audio generated successfully for the word: {word.strip()}")
                self.create_word_buttons(words)
                time.sleep(3)

            except subprocess.CalledProcessError as e:
                self.status_label.config(text=f"Status: Error: {e}")

        url = self.url_entry.get()

        # Check if URL is provided
        if url:
            try:
                python_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'Scripts', 'python.exe')
                subprocess.Popen([python_exe, f"{self.stream_name}_utube_vid_aud.py", url])
                self.status_label.config(text=f"Status: URL processing started successfully for {self.stream_name}.")
                self.root.after(150000, self.moniter_process)
            
                # Get and display channel name
                channel_name = get_channel_name(url)
                if channel_name:
                    self.channel_label.config(text=f"    Channel: {channel_name}")
            
            except Exception as e:
                self.status_label.config(text=f"Status: Error: {e}")

        # Display count of .mp4 videos
        self.update_mp4_count()
        self.current_detected_words = words
        self.create_word_buttons(words)

    def create_word_buttons(self, words):
        # Clear existing word buttons (optional if refreshing)
        if hasattr(self, 'word_buttons'):
            for btn in self.word_buttons:
                btn.destroy()

        self.word_buttons = []
        base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
        
        for idx, word in enumerate(words):
            word = word.strip()
            if not word:
                continue

            folder_path = os.path.join(base_path, f'{self.stream_name}_detection', word)
            mp4_count = 0
            if os.path.exists(folder_path):
                mp4_count = len([f for f in os.listdir(folder_path) if f.endswith('.mp4')])

            button = ttk.Button(
                self.stream_frame,
                text=f"▶ {word} ({mp4_count})",
                command=lambda w=word: self.play_all_word_videos(w),
                style="Word.TButton"
            )
            button.grid(row=6 + idx, column=1, columnspan=2, sticky="w", pady=2)
            self.word_buttons.append(button)

    def play_all_word_videos(self, word):
        base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
        folder_path = os.path.join(base_path, f'{self.stream_name}_detection', word)
        
        if not os.path.exists(folder_path):
            self.status_label.config(text=f"Status: No folder found for '{word}'")
            return

        mp4_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.mp4'))
        if not mp4_files:
            self.status_label.config(text=f"Status: No videos found for '{word}'")
            return

        self.status_label.config(text=f"Status: Playing all videos for '{word}'")

        for video in mp4_files:
            video_path = os.path.join(folder_path, video)
            try:
                # Use xdg-open on Linux or start on Windows
                if os.name == 'nt':  # Windows
                    os.startfile(video_path)
                else:  # Linux
                    subprocess.run(["xdg-open", video_path])
            except Exception as e:
                self.status_label.config(text=f"Error playing: {video} | {e}")
                break

    def get_mp4_count(self):
        base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
        folder_path = os.path.join(base_path, f'{self.stream_name}_detection')
        try:
            total_count = 0
            if os.path.exists(folder_path):
                for word_folder in os.listdir(folder_path):
                    word_path = os.path.join(folder_path, word_folder)
                    if os.path.isdir(word_path):
                        mp4_files = [f for f in os.listdir(word_path) if f.endswith('.mp4')]
                        total_count += len(mp4_files)
            return total_count
        except Exception as e:
            print("Error:", e)
            return 0

    def update_mp4_count(self):
        mp4_count = self.get_mp4_count()
        self.mp4_count_label.config(text=f"Detections: {mp4_count}")
        self.root.after(3000, self.update_mp4_count)

    def start_corelation_updated(self):
        # Detector backend: "correlation" (original np.correlate) or "siamese" (trained model).
        # Switch with the VMS_DETECTOR environment variable; defaults to the original.
        detector = os.environ.get("VMS_DETECTOR", "correlation")
        if detector == "siamese":
            script = f"{self.stream_name}_siamese_detect.py"
        else:
            script = f"{self.stream_name}_corelation_updated_v2.py"
        self.status_label.config(text=f"Status: {script} started successfully.")
        python_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'Scripts', 'python.exe')
        return subprocess.Popen([python_exe, script])

    def moniter_process(self):
        self.monitoring_active = True
        
        def check_process():
            process = self.start_corelation_updated()
            while self.monitoring_active:
                retcode = process.poll()
                if retcode is not None:
                    print(f"Process has died with exit code {retcode}. Restarting...")
                    process = self.start_corelation_updated()
                    if not self.monitoring_active:
                        break
                time.sleep(30)

        monitoring_thread = threading.Thread(target=check_process, daemon=True)
        monitoring_thread.start()

    def stop_processing(self):
        self.action_button.config(text="Start", style="Stream.TButton")
        self.monitoring_active = False
        self.status_label.config(text=f"Status: {self.stream_name} stopped.")

        base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))

        # Kill the subprocess forcefully
        if os.name == 'nt':  # Windows
            subprocess.run(["taskkill", "/F", "/IM", f"{self.stream_name}_hifigan.py"], stderr=subprocess.DEVNULL)
            subprocess.run(["taskkill", "/F", "/IM", f"{self.stream_name}_utube_vid_aud.py"], stderr=subprocess.DEVNULL)
            subprocess.run(["taskkill", "/F", "/IM", f"{self.stream_name}_corelation_updated_v2.py"], stderr=subprocess.DEVNULL)
        else:  # Linux
            subprocess.run(["pkill", "-f", f"{self.stream_name}_hifigan.py"])
            subprocess.run(["pkill", "-f", f"{self.stream_name}_utube_vid_aud.py"])
            subprocess.run(["pkill", "-f", f"{self.stream_name}_corelation_updated_v2.py"])

        # Delete files in videos, audios, and searchword1 folders
        try:
            for folder in [f'{self.stream_name}audios', f'{self.stream_name}videos', f'{self.stream_name}_searchword1']:
                path = os.path.join(base_path, folder)
                if os.path.exists(path):
                    shutil.rmtree(path)
                    os.makedirs(path)
                    print(f'Directory Created for {folder}')
        except Exception as e:
            self.status_label.config(text=f"Status: Error while deleting files: {e}")

    def open_folder(self):
        base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
        if self.stream_name == "Stream1":
            path = os.path.join(base_path, 'Stream1_detection')
        elif self.stream_name == "Stream2":
            path = os.path.join(base_path, 'Stream2_detection')
        else:
            return

        try:
            if os.name == 'nt':  # Windows
                os.startfile(path)
            else:  # Linux
                subprocess.run(['xdg-open', path])
        except Exception as e:
            self.status_label.config(text=f"Status: Error opening folder: {e}")

def toggle_stream2_display(stream2_frame, toggle_button):
    if stream2_frame.winfo_ismapped():
        stream2_frame.grid_remove()
        toggle_button.config(text="+")
    else:
        stream2_frame.grid()
        toggle_button.config(text="-")

def run_stream(stream, root, toggle_button):
    stream_instance = TextToSpeechStream(root, stream["name"], stream["row"])
    if stream["name"] == "Stream2":
        stream_instance.stream_frame.grid_remove()
        toggle_button.config(command=lambda: toggle_stream2_display(stream_instance.stream_frame, toggle_button))

def get_channel_name(url):
    try:
        yt = YouTube(url)
        return yt.author
    except Exception as e:
        print("Error:", e)
        return None

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Live Stream Monitoring Stream")

    # Define stream parameters
    stream1_params = {"name": "Stream1", "row": 0}
    stream2_params = {"name": "Stream2", "row": 1}

    # Create toggle button for Stream 2
    toggle_button = ttk.Button(root, text="+")
    toggle_button.grid(row=2, column=0, columnspan=4, pady=10)

    # Create frames for streams
    stream1_frame = ttk.Frame(root, padding="20", style="Stream.TFrame")
    stream1_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    stream2_frame = ttk.Frame(root, padding="20", style="Stream.TFrame")
    stream2_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

    # Create threads for each stream
    thread1 = threading.Thread(target=run_stream, args=(stream1_params, root, toggle_button))
    thread2 = threading.Thread(target=run_stream, args=(stream2_params, root, toggle_button))

    # Start the threads
    thread1.start()
    thread2.start()

    # Style for the buttons and frames
    style = ttk.Style()
    style.configure("Stream.TFrame", background="#ECEFF1")
    style.configure("Stream.TLabel", foreground="#37474F", background="#ECEFF1")
    style.configure("Stream.TButton", font=("Helvetica", 10), background="#03A9F4", foreground="#FFFFFF", padding=5)
    style.configure("Running.TButton", font=("Helvetica", 10), background="#F44336", foreground="#FFFFFF", padding=5)
    style.configure("Instruction.TLabel", foreground="#757575", background="#ECEFF1", font=("Helvetica", 8))

    # Main loop for the GUI
    root.mainloop()
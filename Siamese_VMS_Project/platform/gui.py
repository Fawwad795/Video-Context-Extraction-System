"""Siamese KWS live monitoring platform - GUI.

Modeled on the original VMS GUI (Correlation_VMS_Project/vms.py), with
downloading and detection started independently:

  Start Download   - begins pulling stream chunks immediately (URL only);
                     the keyword can still be blank or changed.
  Start Detection  - sends the keyword to the running worker (starting it
                     first if needed), which builds the keyword artifacts
                     and begins scanning - including the chunks that
                     accumulated while the keyword was being decided.
  Finish           - stops everything.

Chunks containing a keyword detection are saved to
platform/data/detections/<keyword>/ (video + audio + JSON record), all
other chunks are deleted after analysis so the run never accumulates data.

Everything the platform produces lives under platform/data/ - the research
folders (audios/, keywords/, logs/, videos/) of the Siamese project are
never touched.

Run:  python platform/gui.py
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

PLATFORM_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PLATFORM_DIR, "data")
WORKER = os.path.join(PLATFORM_DIR, "live_worker.py")


class PlatformGUI:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.reader = None
        self.lines = queue.Queue()
        self.session_log = None
        self.create_widgets()
        self.root.after(100, self.drain_output)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI
    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="20", style="Stream.TFrame")
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        # Columns 0/2 (labels) get no weight so they stay snug against their
        # entries; columns 1/3 (entries) absorb all extra width - including
        # the width the row-3 button strip needs beyond columns 0-2's own
        # content, which Tk would otherwise split evenly across 0/1/2
        # (its default when none of a spanning widget's columns have
        # weight), visibly shoving the right-aligned labels rightward.
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)
        frame.columnconfigure(3, weight=2)
        frame.rowconfigure(6, weight=1)

        ttk.Label(frame, text="Siamese Keyword Spotting - Live Stream Monitor",
                  font=("Helvetica", 15, "bold"), style="Stream.TLabel"
                  ).grid(row=0, column=0, columnspan=4, pady=(0, 12), sticky="w")

        ttk.Label(frame, text="Enter a single keyword",
                  style="Instruction.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(frame, text="Input live stream YouTube URL only",
                  style="Instruction.TLabel").grid(row=1, column=3, sticky="w")

        ttk.Label(frame, text="Search Word:", style="Stream.TLabel"
                  ).grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.word_entry = ttk.Entry(frame, width=24)
        self.word_entry.grid(row=2, column=1, padx=5, pady=5, sticky="we")

        ttk.Label(frame, text="URL:", style="Stream.TLabel"
                  ).grid(row=2, column=2, padx=5, pady=5, sticky="e")
        self.url_entry = ttk.Entry(frame, width=46)
        self.url_entry.grid(row=2, column=3, padx=5, pady=5, sticky="we")

        buttons = ttk.Frame(frame, style="Stream.TFrame")
        buttons.grid(row=3, column=0, columnspan=3, pady=8, sticky="w")
        self.download_button = ttk.Button(buttons, text="Start Download",
                                          command=self.start_download,
                                          style="Stream.TButton")
        self.download_button.pack(side="left", padx=(0, 6))
        self.detect_button = ttk.Button(buttons, text="Start Detection",
                                        command=self.start_detection,
                                        style="Stream.TButton")
        self.detect_button.pack(side="left", padx=(0, 6))
        self.finish_button = ttk.Button(buttons, text="Finish",
                                        command=self.stop, state="disabled",
                                        style="Stream.TButton")
        self.finish_button.pack(side="left", padx=(0, 6))
        self.view_button = ttk.Button(buttons, text="View Detections",
                                      command=self.open_detections,
                                      style="Stream.TButton")
        self.view_button.pack(side="left", padx=(0, 6))
        self.clear_button = ttk.Button(buttons, text="Clear Console",
                                       command=self.clear_console,
                                       style="Stream.TButton")
        self.clear_button.pack(side="left")
        self.queue_label = ttk.Label(frame, text="Backlog: 0",
                                     style="Stream.TLabel")
        self.queue_label.grid(row=3, column=3, pady=8, sticky="w")
        self.count_label = ttk.Label(frame, text="Detections: 0",
                                     style="Stream.TLabel")
        self.count_label.grid(row=3, column=3, pady=8, sticky="e")

        self.phase_label = ttk.Label(frame, text="Phase: idle",
                                     font=("Helvetica", 10, "bold"),
                                     style="Stream.TLabel")
        self.phase_label.grid(row=4, column=0, columnspan=4, sticky="w")
        self.status_label = ttk.Label(frame, text="Status: ready",
                                      style="Stream.TLabel", wraplength=760)
        self.status_label.grid(row=5, column=0, columnspan=4, sticky="w")

        log_frame = ttk.Frame(frame)
        log_frame.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=18, width=100, state="disabled",
                                bg="#263238", fg="#B0BEC5", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scroll.set)

        self.update_count()

    # ------------------------------------------------------------- actions
    def worker_running(self):
        return self.process is not None and self.process.poll() is None

    def launch_worker(self, url):
        """Start the worker in download-only mode (keyword sent later)."""
        os.makedirs(os.path.join(DATA_ROOT, "logs"), exist_ok=True)
        session_path = os.path.join(
            DATA_ROOT, "logs", f"session_{datetime.now():%Y%m%d_%H%M%S}.log")
        self.session_log = open(session_path, "a", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        self.process = subprocess.Popen(
            [sys.executable, "-u", WORKER, "--url", url, "--root", DATA_ROOT],
            cwd=PLATFORM_DIR, env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.reader = threading.Thread(target=self.read_output, daemon=True)
        self.reader.start()

        self.download_button.config(state="disabled")
        self.finish_button.config(state="normal")
        self.append_log(f"--- download started: url={url} ---")

    def start_download(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Input", "Enter a live stream URL.")
            return
        if self.worker_running():
            return
        self.launch_worker(url)
        self.phase_label.config(text="Phase: download")
        self.status_label.config(
            text="Status: downloading chunks - enter a keyword and press "
                 "Start Detection when ready.")

    def start_detection(self):
        keyword = self.word_entry.get().strip().lower()
        if not keyword or " " in keyword:
            messagebox.showwarning("Input", "Enter a single keyword (no spaces).")
            return
        if not self.worker_running():
            # One-click flow: start downloading and detecting together.
            url = self.url_entry.get().strip()
            if not url:
                messagebox.showwarning("Input", "Enter a live stream URL.")
                return
            self.launch_worker(url)

        self.keyword = keyword
        try:
            self.process.stdin.write(f"KEYWORD {keyword}\n")
            self.process.stdin.flush()
        except OSError as e:
            messagebox.showerror("Worker", f"Could not send keyword: {e}")
            return

        self.detect_button.config(state="disabled")
        self.word_entry.config(state="disabled")
        self.phase_label.config(text="Phase: setup")
        self.status_label.config(
            text=f"Status: keyword '{keyword}' sent - building keyword "
                 "artifacts (first run for a keyword can take a while)")
        self.append_log(f"--- detection started: keyword='{keyword}' ---")

    def stop(self):
        if self.worker_running():
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.process.terminate()
        self.process = None
        self.reset_buttons()
        self.phase_label.config(text="Phase: idle")
        self.status_label.config(text="Status: stopped.")
        self.append_log("--- session stopped ---")
        if self.session_log:
            self.session_log.close()
            self.session_log = None

    def reset_buttons(self):
        self.download_button.config(state="normal")
        self.detect_button.config(state="normal")
        self.finish_button.config(state="disabled")
        self.word_entry.config(state="normal")

    def clear_console(self):
        """Clears only the on-screen console - the session log file on disk
        (platform/data/logs/session_*.log) keeps the full record either way."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def open_detections(self):
        keyword = self.word_entry.get().strip().lower()
        path = os.path.join(DATA_ROOT, "detections")
        if keyword and os.path.isdir(os.path.join(path, keyword)):
            path = os.path.join(path, keyword)
        os.makedirs(path, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.run(["xdg-open", path])

    # -------------------------------------------------------------- output
    def read_output(self):
        proc = self.process
        for line in proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(f"@@EXIT {proc.wait()}")

    def drain_output(self):
        try:
            while True:
                line = self.lines.get_nowait()
                self.handle_line(line)
        except queue.Empty:
            pass
        self.root.after(100, self.drain_output)

    def handle_line(self, line):
        if self.session_log:
            self.session_log.write(line + "\n")
            self.session_log.flush()
        if line.startswith("@@PHASE "):
            self.phase_label.config(text=f"Phase: {line[8:]}")
        elif line.startswith("@@STATUS "):
            self.status_label.config(text=f"Status: {line[9:]}")
            self.append_log(line[9:])
        elif line.startswith("@@DETECT "):
            self.count_label.config(text=f"Detections: {line[9:]}")
        elif line.startswith("@@QUEUE "):
            self.queue_label.config(text=f"Backlog: {line[8:]}")
        elif line.startswith("@@EXIT "):
            code = line[7:]
            if self.process is not None:  # died on its own, not via Stop
                self.append_log(f"worker exited (code {code})")
                self.status_label.config(
                    text=f"Status: worker exited (code {code}) - see log above.")
                self.process = None
                self.reset_buttons()
                self.phase_label.config(text="Phase: idle")
        else:
            self.append_log(line)

    def append_log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        # Keep the widget bounded for long sessions
        if int(self.log_text.index("end-1c").split(".")[0]) > 2000:
            self.log_text.delete("1.0", "500.0")
        self.log_text.config(state="disabled")

    def update_count(self):
        keyword = getattr(self, "keyword", None) or self.word_entry.get().strip().lower()
        det_dir = os.path.join(DATA_ROOT, "detections", keyword) if keyword else None
        if det_dir and os.path.isdir(det_dir) and not (
                self.process and self.process.poll() is None):
            n = len([f for f in os.listdir(det_dir) if f.endswith(".mp4")])
            self.count_label.config(text=f"Detections: {n} (saved)")
        self.root.after(3000, self.update_count)

    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Siamese KWS - Live Stream Monitoring Platform")
    root.geometry("860x640")

    style = ttk.Style()
    style.configure("Stream.TFrame", background="#ECEFF1")
    style.configure("Stream.TLabel", foreground="#37474F", background="#ECEFF1")
    style.configure("Stream.TButton", font=("Helvetica", 10), padding=5)
    style.configure("Running.TButton", font=("Helvetica", 10), padding=5)
    style.configure("Instruction.TLabel", foreground="#757575",
                    background="#ECEFF1", font=("Helvetica", 8))

    PlatformGUI(root)
    root.mainloop()

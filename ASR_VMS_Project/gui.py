import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import subprocess
from config import WORK_DIR

class VMS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASR Video Monitoring System")
        self.root.geometry("800x600")
        
        self.process = None
        
        # Main Frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # URL
        ttk.Label(main_frame, text="YouTube Live URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(main_frame, textvariable=self.url_var, width=60)
        self.url_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        # Keywords
        ttk.Label(main_frame, text="Keywords (comma separated):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.kw_var = tk.StringVar()
        self.kw_entry = ttk.Entry(main_frame, textvariable=self.kw_var, width=60)
        self.kw_entry.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        # Controls
        self.start_btn = ttk.Button(main_frame, text="Start Detection", command=self.start_detection)
        self.start_btn.grid(row=2, column=0, pady=20)
        
        self.stop_btn = ttk.Button(main_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.grid(row=2, column=1, pady=20)
        
        self.view_btn = ttk.Button(main_frame, text="View Output Folder", command=self.open_folder)
        self.view_btn.grid(row=2, column=2, pady=20)
        
        # Log Box
        ttk.Label(main_frame, text="Status Log:").grid(row=3, column=0, sticky=tk.W)
        self.log_text = tk.Text(main_frame, height=20, width=80)
        self.log_text.grid(row=4, column=0, columnspan=3, pady=5)
        
        self.log("GUI Initialized. Ready to start ASR detection pipeline.")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def start_detection(self):
        url = self.url_var.get().strip()
        keywords = self.kw_var.get().strip()
        
        if not url or not keywords:
            messagebox.showerror("Error", "Please provide both URL and Keywords.")
            return
            
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.log(f"Starting pipeline for URL: {url}")
        self.log(f"Keywords: {keywords}")
        
        # Start backend pipeline thread
        def run_pipeline():
            # In production, we'd launch detector.py via subprocess or import it.
            # Here we just launch it via subprocess
            cmd = ["python", "detector.py", "--url", url, "--keywords", keywords]
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            for line in self.process.stdout:
                self.root.after(0, self.log, line.strip())
                
            self.root.after(0, self.pipeline_finished)
            
        threading.Thread(target=run_pipeline, daemon=True).start()

    def stop_detection(self):
        if self.process:
            self.process.terminate()
            self.process = None
        self.log("Detection stopped by user.")
        self.pipeline_finished()

    def pipeline_finished(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
    def open_folder(self):
        if os.name == 'nt':
            os.startfile(WORK_DIR)
        else:
            subprocess.run(['xdg-open', WORK_DIR])

if __name__ == "__main__":
    root = tk.Tk()
    app = VMS_GUI(root)
    root.mainloop()

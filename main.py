import os
import time
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from tkinter import filedialog
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

# Load environment variables
load_dotenv()

# Constants
MAX_FILE_SIZE_MB = 25  # OpenAI Whisper API limit
BYTES_TO_MB = 1024 * 1024
TARGET_CHUNK_SIZE_MB = 20  # Target size for each chunk (slightly below max to be safe)

class AudioChunk:
    def __init__(self, file_path, chunk_number, total_chunks):
        self.file_path = file_path
        self.chunk_number = chunk_number
        self.total_chunks = total_chunks
        self.transcript = ""
        self.notes = ""

class PitchRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Pitch Call Recorder and Transcriber")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Set application icon (if available)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        # Initialize OpenAI client with API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            messagebox.showerror("Error", "OpenAI API key not found. Please set it in the .env file.")
            root.destroy()
            return
            
        self.client = OpenAI(api_key=api_key)
        self.recording = False
        self.audio_data = []
        self.sample_rate = 44100
        self.channels = 2
        self.current_audio_file = None
        self.audio_chunks = []
        self.animation = None
        
        # Configure style
        self.setup_styles()
        self.setup_gui()
        self.setup_keyboard_shortcuts()
        
    def setup_styles(self):
        """Configure modern styling for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Status.TLabel', font=('Arial', 12), foreground='#34495e')
        style.configure('Success.TLabel', font=('Arial', 12), foreground='#27ae60')
        style.configure('Error.TLabel', font=('Arial', 12), foreground='#e74c3c')
        
        # Configure buttons
        style.configure('Record.TButton', font=('Arial', 12, 'bold'), padding=10)
        style.configure('Action.TButton', font=('Arial', 10), padding=8)
        
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🎤 Pitch Call Recorder & Transcriber", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.status_label = ttk.Label(status_frame, text="Ready to record", style='Status.TLabel')
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.file_size_label = ttk.Label(status_frame, text="", style='Status.TLabel')
        self.file_size_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Audio visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Audio Visualization", padding="10")
        viz_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.fig = Figure(figsize=(8, 2), facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('Time (s)')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Recording controls
        record_frame = ttk.Frame(controls_frame)
        record_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.record_button = ttk.Button(
            record_frame, 
            text="🎙️ Start Recording", 
            command=self.toggle_recording,
            style='Record.TButton'
        )
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.transcribe_button = ttk.Button(
            record_frame, 
            text="📝 Transcribe", 
            command=self.transcribe_current, 
            state='disabled',
            style='Action.TButton'
        )
        self.transcribe_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(
            record_frame, 
            text="🗑️ Clear", 
            command=self.clear_all,
            style='Action.TButton'
        )
        self.clear_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(controls_frame, mode='indeterminate', length=300)
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        # Transcript tab
        transcript_frame = ttk.Frame(notebook)
        notebook.add(transcript_frame, text='📄 Transcript')
        
        transcript_header = ttk.Frame(transcript_frame)
        transcript_header.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(transcript_header, text="Transcription Results:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Button(transcript_header, text="Copy", command=self.copy_transcript).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(transcript_header, text="Clear", command=self.clear_transcript).pack(side=tk.RIGHT)
        
        self.transcript_text = scrolledtext.ScrolledText(
            transcript_frame, 
            wrap=tk.WORD, 
            height=12,
            font=('Consolas', 10),
            bg='#ffffff',
            fg='#2c3e50'
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # Notes tab
        notes_frame = ttk.Frame(notebook)
        notebook.add(notes_frame, text='📋 Investor Notes')
        
        notes_header = ttk.Frame(notes_frame)
        notes_header.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(notes_header, text="Generated Notes:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Button(notes_header, text="Copy", command=self.copy_notes).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(notes_header, text="Clear", command=self.clear_notes).pack(side=tk.RIGHT)
        
        self.notes_text = scrolledtext.ScrolledText(
            notes_frame, 
            wrap=tk.WORD, 
            height=12,
            font=('Consolas', 10),
            bg='#ffffff',
            fg='#2c3e50'
        )
        self.notes_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # Save buttons
        save_frame = ttk.LabelFrame(main_frame, text="Save Options", padding="10")
        save_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        ttk.Button(save_frame, text="💾 Save Transcript", command=self.save_transcript).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(save_frame, text="💾 Save Notes", command=self.save_notes).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(save_frame, text="💾 Save Both", command=self.save_both).pack(side=tk.LEFT)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common actions"""
        self.root.bind('<Control-r>', lambda e: self.toggle_recording())
        self.root.bind('<Control-t>', lambda e: self.transcribe_current())
        self.root.bind('<Control-s>', lambda e: self.save_both())
        self.root.bind('<Control-c>', lambda e: self.copy_transcript())
        self.root.bind('<Control-n>', lambda e: self.copy_notes())
        self.root.bind('<Escape>', lambda e: self.stop_recording() if self.recording else None)
        
        # Show keyboard shortcuts in status
        shortcuts_text = "Shortcuts: Ctrl+R (Record), Ctrl+T (Transcribe), Ctrl+S (Save), Ctrl+C (Copy), Esc (Stop)"
        shortcuts_label = ttk.Label(self.root, text=shortcuts_text, font=('Arial', 8), foreground='#7f8c8d')
        shortcuts_label.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=15)

    def update_visualization(self, audio_chunk):
        """Update the audio visualization"""
        try:
            if len(audio_chunk) > 0:
                # Handle stereo audio by taking the mean of both channels
                if len(audio_chunk.shape) > 1:
                    audio_data = np.mean(audio_chunk, axis=1)
                else:
                    audio_data = audio_chunk.flatten()
                
                # Calculate RMS for visualization
                rms = np.sqrt(np.mean(audio_data**2))
                
                # Update the plot
                self.ax.clear()
                self.ax.set_ylim(-1, 1)
                self.ax.set_xlim(0, 100)
                self.ax.set_ylabel('Amplitude')
                self.ax.set_xlabel('Time (s)')
                self.ax.grid(True, alpha=0.3)
                
                # Create a simple waveform visualization
                # Ensure x and y have the same dimensions
                time_points = np.linspace(0, 100, len(audio_data))
                
                # Only plot if we have data
                if len(audio_data) > 0:
                    self.ax.plot(time_points, audio_data, color='#3498db', linewidth=0.5)
                
                # Add volume indicator
                volume_color = '#e74c3c' if rms > 0.5 else '#f39c12' if rms > 0.2 else '#27ae60'
                self.ax.axhline(y=rms, color=volume_color, linestyle='--', alpha=0.7, label=f'Volume: {rms:.3f}')
                self.ax.axhline(y=-rms, color=volume_color, linestyle='--', alpha=0.7)
                self.ax.legend()
                
                self.canvas.draw()
        except Exception as e:
            # Silently handle visualization errors to prevent app crashes
            print(f"Visualization error: {e}")

    def get_file_size_mb(self, file_path):
        """Get file size in megabytes"""
        return os.path.getsize(file_path) / BYTES_TO_MB

    def update_file_size_label(self, file_path):
        """Update the file size label with current file size"""
        if file_path and os.path.exists(file_path):
            size_mb = self.get_file_size_mb(file_path)
            color = "red" if size_mb > MAX_FILE_SIZE_MB else "green" if size_mb < MAX_FILE_SIZE_MB * 0.8 else "orange"
            self.file_size_label.configure(
                text=f"File size: {size_mb:.2f} MB (Max: {MAX_FILE_SIZE_MB} MB)",
                foreground=color
            )
        else:
            self.file_size_label.configure(text="")

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.record_button.configure(text="⏹️ Stop Recording")
        self.status_label.configure(text="🎙️ Recording... Press Esc to stop", foreground='#e74c3c')
        self.progress.start()
        
        def record_thread():
            def callback(indata, frames, time, status):
                if status:
                    print(status)
                if self.recording:
                    self.audio_data.append(indata.copy())
                    # Update visualization in main thread
                    self.root.after(0, lambda: self.update_visualization(indata))
            
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=callback):
                while self.recording:
                    time.sleep(0.1)
        
        threading.Thread(target=record_thread, daemon=True).start()

    def stop_recording(self):
        self.recording = False
        self.record_button.configure(text="🎙️ Start Recording")
        self.progress.stop()
        
        if self.audio_data:
            audio_data = np.concatenate(self.audio_data, axis=0)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_audio_file = f"recording_{timestamp}.wav"
            wav.write(self.current_audio_file, self.sample_rate, audio_data)
            self.status_label.configure(text=f"✅ Recording saved as {self.current_audio_file}", foreground='#27ae60')
            self.transcribe_button.configure(state='normal')
            self.update_file_size_label(self.current_audio_file)
            
            # Clear visualization
            self.ax.clear()
            self.ax.set_ylim(-1, 1)
            self.ax.set_xlim(0, 100)
            self.ax.set_ylabel('Amplitude')
            self.ax.set_xlabel('Time (s)')
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
        else:
            self.status_label.configure(text="❌ No recording was made", foreground='#e74c3c')
            self.file_size_label.configure(text="")

    def clear_all(self):
        """Clear all data and reset the interface"""
        if self.recording:
            self.stop_recording()
        
        self.current_audio_file = None
        self.audio_chunks = []
        self.clear_transcript()
        self.clear_notes()
        self.status_label.configure(text="Ready to record", foreground='#34495e')
        self.file_size_label.configure(text="")
        self.transcribe_button.configure(state='disabled')
        
        # Clear visualization
        self.ax.clear()
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlabel('Time (s)')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def clear_transcript(self):
        """Clear the transcript text"""
        self.transcript_text.delete(1.0, tk.END)

    def clear_notes(self):
        """Clear the notes text"""
        self.notes_text.delete(1.0, tk.END)

    def copy_transcript(self):
        """Copy transcript to clipboard"""
        transcript = self.transcript_text.get(1.0, tk.END).strip()
        if transcript:
            self.root.clipboard_clear()
            self.root.clipboard_append(transcript)
            messagebox.showinfo("Copied", "Transcript copied to clipboard!")

    def copy_notes(self):
        """Copy notes to clipboard"""
        notes = self.notes_text.get(1.0, tk.END).strip()
        if notes:
            self.root.clipboard_clear()
            self.root.clipboard_append(notes)
            messagebox.showinfo("Copied", "Notes copied to clipboard!")

    def calculate_chunk_duration(self, file_path):
        """Calculate the appropriate chunk duration to stay under size limit"""
        sample_rate, audio_data = wav.read(file_path)
        bytes_per_sample = audio_data.dtype.itemsize
        bytes_per_second = sample_rate * bytes_per_sample * self.channels
        
        # Calculate how many seconds of audio we can fit in our target chunk size
        max_bytes_per_chunk = TARGET_CHUNK_SIZE_MB * BYTES_TO_MB
        chunk_duration = max_bytes_per_chunk / bytes_per_second
        
        return int(chunk_duration)

    def split_audio_file(self, file_path):
        """Split audio file into chunks if it's too large"""
        sample_rate, audio_data = wav.read(file_path)
        chunk_duration = self.calculate_chunk_duration(file_path)
        
        # Calculate total duration and number of chunks needed
        duration_seconds = len(audio_data) / sample_rate
        num_chunks = math.ceil(duration_seconds / chunk_duration)
        
        if num_chunks <= 1:
            return [AudioChunk(file_path, 1, 1)]
        
        chunks = []
        chunk_samples = int(chunk_duration * sample_rate)
        
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min((i + 1) * chunk_samples, len(audio_data))
            chunk_data = audio_data[start_idx:end_idx]
            
            chunk_filename = f"{os.path.splitext(file_path)[0]}_chunk_{i+1}.wav"
            wav.write(chunk_filename, sample_rate, chunk_data)
            
            # Verify chunk size
            chunk_size_mb = os.path.getsize(chunk_filename) / BYTES_TO_MB
            if chunk_size_mb > MAX_FILE_SIZE_MB:
                # If chunk is still too large, try a smaller duration
                os.remove(chunk_filename)
                chunk_duration = int(chunk_duration * 0.8)  # Reduce duration by 20%
                chunk_samples = int(chunk_duration * sample_rate)
                chunk_data = audio_data[start_idx:min(start_idx + chunk_samples, len(audio_data))]
                wav.write(chunk_filename, sample_rate, chunk_data)
            
            chunks.append(AudioChunk(chunk_filename, i+1, num_chunks))
        
        return chunks

    def transcribe_current(self):
        if not self.current_audio_file:
            messagebox.showerror("Error", "No recording available to transcribe")
            return
        
        # Check file size before transcription
        file_size_mb = self.get_file_size_mb(self.current_audio_file)
        if file_size_mb > MAX_FILE_SIZE_MB:
            self.status_label.configure(text="Splitting audio file...")
            self.audio_chunks = self.split_audio_file(self.current_audio_file)
            
            # Verify all chunks are under size limit
            for chunk in self.audio_chunks:
                chunk_size = os.path.getsize(chunk.file_path) / BYTES_TO_MB
                if chunk_size > MAX_FILE_SIZE_MB:
                    messagebox.showerror(
                        "Error",
                        f"Chunk {chunk.chunk_number} is still too large ({chunk_size:.2f} MB). Please try recording with lower quality settings."
                    )
                    return
            
            self.status_label.configure(text=f"Split into {len(self.audio_chunks)} chunks")
        
        self.status_label.configure(text="Transcribing...")
        self.progress.start()
        self.transcribe_button.configure(state='disabled')
        
        def transcribe_thread():
            try:
                if not self.audio_chunks:
                    self.audio_chunks = [AudioChunk(self.current_audio_file, 1, 1)]
                
                all_transcripts = []
                
                # Step 1: Transcribe all audio chunks
                for i, chunk in enumerate(self.audio_chunks):
                    self.root.after(0, lambda i=i: self.status_label.configure(
                        text=f"Transcribing chunk {i+1}/{len(self.audio_chunks)}..."
                    ))
                    
                    with open(chunk.file_path, "rb") as file:
                        # Enhanced transcription parameters
                        transcript = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=file,
                            response_format="verbose_json",
                            language="en",
                            temperature=0.2,  # Lower temperature for more accurate transcription
                            prompt="This is a business pitch call. The audio may contain technical terms, business jargon, and financial numbers. Please transcribe accurately with proper punctuation and formatting."
                        )
                        
                        # Process the transcript to improve quality
                        processed_text = self.process_transcript(transcript.text)
                        chunk.transcript = processed_text
                        all_transcripts.append(processed_text)
                
                # Step 2: Combine all transcripts
                self.root.after(0, lambda: self.status_label.configure(text="Combining transcripts..."))
                combined_transcript = self.combine_transcripts(all_transcripts)
                
                # Step 3: Generate notes from the combined transcript
                self.root.after(0, lambda: self.status_label.configure(text="Generating investor notes..."))
                notes = self.generate_notes_from_transcript(combined_transcript)
                
                # Update UI with results
                self.root.after(0, lambda: self.update_transcript(combined_transcript))
                self.root.after(0, lambda: self.update_notes(notes))
                
                # Clean up chunk files
                for chunk in self.audio_chunks:
                    if chunk.file_path != self.current_audio_file:
                        try:
                            os.remove(chunk.file_path)
                        except:
                            pass
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror("Error", f"Transcription failed: {error_msg}"))
            finally:
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.status_label.configure(text="Transcription complete"))
                self.root.after(0, lambda: self.transcribe_button.configure(state='normal'))
                self.audio_chunks = []
        
        threading.Thread(target=transcribe_thread, daemon=True).start()

    def process_transcript(self, text):
        """Process and enhance the transcript quality"""
        try:
            # Use GPT-4 to improve the transcript
            prompt = f"""Please improve this transcript from a business pitch call. Make the following enhancements:
            1. Fix any transcription errors
            2. Add proper punctuation and formatting
            3. Correct any technical terms or business jargon
            4. Format numbers and percentages properly
            5. Break into clear paragraphs
            6. Maintain the original meaning and context

            Original transcript:
            {text}"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert transcription editor specializing in business and technical content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=4000  # Limit to prevent overflow
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return text  # Return original text if processing fails

    def combine_transcripts(self, transcripts):
        """Combine multiple transcripts into a coherent document"""
        try:
            combined_text = "\n\n".join(transcripts)
            
            # If combined text is too long, truncate it
            if len(combined_text) > 8000:
                combined_text = combined_text[:8000] + "\n\n[Transcript truncated due to length]"
            
            prompt = f"""Please combine these sections of a business pitch call transcript into a single coherent document. Make the following improvements:
            1. Ensure smooth transitions between sections
            2. Remove any redundancies
            3. Maintain consistent formatting
            4. Preserve the chronological order
            5. Keep all important details and context

            Transcript sections to combine:
            {combined_text}"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert transcription editor specializing in business content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000  # Limit to prevent overflow
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return "\n\n".join(transcripts)  # Return simple concatenation if processing fails

    def generate_notes_from_transcript(self, transcript):
        """Generate comprehensive investor notes from the complete transcript"""
        try:
            # Truncate transcript if too long to prevent input overflow
            if len(transcript) > 6000:
                transcript = transcript[:6000] + "\n\n[Transcript truncated for note generation]"
            
            prompt = f"""Analyze this complete pitch call transcript and create comprehensive investor-ready notes. 
            Structure the notes with clear sections and bullet points. Include:

            1. Key Points - Main takeaways and highlights
            2. Investment Ask - Funding amount, terms, and valuation
            3. Market Opportunity - Market size, growth potential, target market
            4. Competitive Advantage - Unique selling propositions, moats
            5. Financial Highlights - Current revenue, growth, key metrics
            6. Team Members - Key personnel and their roles
            7. Experience - Team's relevant background and track record
            8. Use of Funds - How the investment will be allocated
            9. Financial Projections - Future revenue and growth forecasts
            10. Exit Strategy - Potential exit scenarios and timelines
            11. Risks - Key risks and mitigation strategies
            12. Conclusion - Summary and next steps

            Make the notes professional, concise, and actionable for investors.

            Transcript:
            {transcript}"""

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert investment analyst. Create clear, comprehensive, and professional investor notes from the given transcript."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=3000  # Limit to prevent overflow
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating notes: {str(e)}"

    def update_notes(self, notes):
        """Update the notes text widget with the combined notes"""
        self.notes_text.delete(1.0, tk.END)
        self.notes_text.insert(tk.END, notes)

    def update_transcript(self, transcript):
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.insert(tk.END, transcript)

    def save_transcript(self):
        if not self.transcript_text.get(1.0, tk.END).strip():
            messagebox.showerror("Error", "No transcript to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.transcript_text.get(1.0, tk.END))
            messagebox.showinfo("Success", "Transcript saved successfully")

    def save_notes(self):
        if not self.notes_text.get(1.0, tk.END).strip():
            messagebox.showerror("Error", "No notes to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.notes_text.get(1.0, tk.END))
            messagebox.showinfo("Success", "Notes saved successfully")

    def save_both(self):
        """Save both transcript and notes"""
        if not self.transcript_text.get(1.0, tk.END).strip() and not self.notes_text.get(1.0, tk.END).strip():
            messagebox.showerror("Error", "No transcript or notes to save")
            return
        
        # Save transcript if available
        if self.transcript_text.get(1.0, tk.END).strip():
            self.save_transcript()
        
        # Save notes if available
        if self.notes_text.get(1.0, tk.END).strip():
            self.save_notes()

def main():
    root = tk.Tk()
    app = PitchRecorderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
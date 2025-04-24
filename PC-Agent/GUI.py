import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading
import sys
import queue  # Used for inter-thread communication of output data
import time   # Import time for timestamping log file and waiting
import os     # Import os for directory handling
import re     # Import regular expressions for filtering

class RunProcessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PC Agent GUI")
        self.geometry("900x700")

        # Define variables for parameters
        self.instruction = tk.StringVar(value="Using Edge, add a adidas hat under $20 to cart in amazon.")
        self.icon_caption = tk.BooleanVar(value=False)
        self.use_som = tk.BooleanVar(value=True)
        self.draw_text_box = tk.BooleanVar(value=False)
        self.pc_type = tk.StringVar(value="windows")
        self.enable_reflection = tk.BooleanVar(value=True)
        self.enable_memory = tk.BooleanVar(value=True)
        self.enable_eval = tk.BooleanVar(value=True)
        self.enable_reward = tk.BooleanVar(value=False)
        # ============================================================
        # CHANGED DEFAULT MODEL TO GEMINI
        # ============================================================
        self.model_backend = tk.StringVar(value="gemini") # Default model changed
        # ============================================================

        self.build_gui()
        self.process = None
        self.process_thread = None
        self.output_queue = queue.Queue()
        self.log_file = None

        # ============================================================
        # UPDATED FILTER REGEX PATTERN (v4)
        # ============================================================
        # (Filter pattern remains the same as the previous version)
        self._filter_pattern = re.compile(
            # --- Original Single-line Patterns ---
            r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]\d+\s+-\s+\w+\s+-|"
            r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+:\s+[IWEF]\s+tensorflow|"
            r"WARNING:tensorflow:|"
            r"SupervisionWarnings:|"

            # --- New Specific Lines ---
            r"^final text_encoder_type:|(?i)"
            r"^\s*To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags\.$|"

            # --- Enhanced Patterns for TF Update Instructions (v4) ---
            r"Instructions for updating:$|"
            r"^\s*tf\.py_func is deprecated in TF V2\. Instead, there are two|"
            r"^\s*options available in V2\.?|(?i)"
            r"^\s*- tf\.py_function takes a python function which manipulates tf eager|(?i)"
            r"^\s*tensors instead of numpy arrays\. It's easy to convert a tf eager tensor to|(?i)"
            r"^\s*an ndarray \(just call tensor\.numpy\(\)\) but having access to eager tensors|(?i)"
            r"^\s*means `tf\.py_function`s can use accelerators such as GPUs as well as|(?i)"
            r"^\s*being differentiable using a gradient tape\.?|(?i)"
            r"^\s*- tf\.numpy_function maintains the semantics of the deprecated tf\.py_func|(?i)"
            r"^\s*\(it is not differentiable, and manipulates numpy arrays\)\. It drops the|(?i)"
            r"^\s*stateful argument making all functions stateful\.?|(?i)"
            r"^\s*Use `tf\.config\.list_physical_devices\('GPU'\)` instead\.?$"
            r")"
        )
        # ============================================================

        # Start polling the output queue
        self.poll_output_queue()

    def build_gui(self):
        # --- GUI layout --- (Same as before)
        # First row: Instruction input
        ttk.Label(self, text="Instruction:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self, textvariable=self.instruction, width=80).grid(row=0, column=1, columnspan=4, sticky="ew", padx=5, pady=5)
        self.grid_columnconfigure(1, weight=1)

        # Second row: Feature options
        ttk.Checkbutton(self, text="Enable Icon Caption", variable=self.icon_caption).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable SOM", variable=self.use_som).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable Draw Text Box", variable=self.draw_text_box).grid(row=1, column=2, sticky="w", padx=5, pady=5)

        # Third row: System type and Model selection
        ttk.Label(self, text="PC Type:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        pc_frame = ttk.Frame(self)
        pc_frame.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(pc_frame, text="Windows", variable=self.pc_type, value="windows").pack(side=tk.LEFT)
        ttk.Radiobutton(pc_frame, text="Mac", variable=self.pc_type, value="mac").pack(side=tk.LEFT, padx=10)
        ttk.Label(self, text="Model:").grid(row=2, column=2, sticky="e", padx=5, pady=5)
        # The Combobox will now show "gemini" by default due to the variable change
        ttk.Combobox(self, textvariable=self.model_backend, values=["gpt", "gemini"], state="readonly", width=10).grid(row=2, column=3, sticky="w", padx=5, pady=5)

        # Fourth row: Feature switches
        ttk.Checkbutton(self, text="Action Scaling", variable=self.enable_reflection).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Planning Scaling", variable=self.enable_eval).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable Memory", variable=self.enable_memory).grid(row=3, column=2, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Offline Reward", variable=self.enable_reward).grid(row=3, column=3, sticky="w", padx=5, pady=5)

        # Fifth row: Start and Stop buttons
        button_frame = ttk.Frame(self)
        button_frame.grid(row=4, column=0, columnspan=4, pady=10)
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_process)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_process, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Sixth row: Output log scrolled text box
        ttk.Label(self, text="Process Output:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.output_text = ScrolledText(self, wrap="word", width=100, height=30)
        self.output_text.grid(row=6, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)
        self.grid_rowconfigure(6, weight=1)

    def start_process(self):
        # --- start_process method --- (Same as before)
        if self.process and self.process.poll() is None:
            self.output_text.insert(tk.END, "Process already running.\n")
            return

        # Determine wrapper script and base command
        if sys.platform.startswith("win"):
            wrapper_script = "run_with_env.bat"
            base_cmd = ["cmd", "/c", wrapper_script]
        else:
            wrapper_script = "run_with_env.sh"
            base_cmd = ["bash", wrapper_script]

        # Construct arguments for run.py
        run_py_args = [
            "--instruction", self.instruction.get(),
            "--icon_caption", "1" if self.icon_caption.get() else "0",
            "--use_som", "1" if self.use_som.get() else "0",
            "--draw_text_box", "1" if self.draw_text_box.get() else "0",
            "--pc_type", self.pc_type.get(),
            "--direct_print"
        ]
        if not self.enable_reflection.get():
            run_py_args.append("--disable_reflection")
        if not self.enable_memory.get():
            run_py_args.append("--disable_memory")
        if not self.enable_eval.get():
            run_py_args.append("--disable_eval")
        if self.enable_eval.get() and self.enable_reward.get():
             run_py_args.append("--enable_reward")
        # The selected value (defaulting to gemini) will be passed
        run_py_args.extend(["--model_backend", self.model_backend.get()])
        cmd = base_cmd + run_py_args

        # File Logging Setup
        log_dir = "log"
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
             self.output_text.insert(tk.END, f"Error creating log directory '{log_dir}': {e}\n")
             log_dir = "." # Fallback

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"process_output_{timestamp}.log")

        try:
            # Open log file and write header
            self.log_file = open(log_filename, "w", encoding='utf-8', buffering=1)
            status_header = f"""--- Initial Configuration ---
Timestamp: {timestamp}
Instruction: {self.instruction.get()}
PC Type: {self.pc_type.get()}
Model: {self.model_backend.get()}

Checkbox States:
  Action Scaling (enable_reflection): {self.enable_reflection.get()}
  Enable Memory (enable_memory):      {self.enable_memory.get()}
  Planning Scaling (enable_eval):   {self.enable_eval.get()}
  Offline Reward (enable_reward):   {self.enable_reward.get()}
  --- Other Options ---
  Enable Icon Caption: {self.icon_caption.get()}
  Enable SOM:          {self.use_som.get()}
  Enable Draw Text Box:{self.draw_text_box.get()}
--------------------------

Command Executed:
{' '.join(cmd)}

--- Process Output Starts Below ---
"""
            self.log_file.write(status_header)

        except IOError as e:
            self.output_text.insert(tk.END, f"Error opening log file {log_filename}: {e}\n")
            self.log_file = None
            return # Stop if log file fails

        # Start Process
        self.output_text.insert(tk.END, f"Saving output also to: {log_filename}\n")
        self.output_text.insert(tk.END, "Starting process with command:\n" + " ".join(cmd) + "\n")
        self.output_text.see(tk.END)
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        try:
            # Start the subprocess
            creation_flags = 0
            if sys.platform.startswith("win"):
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

            self.process = subprocess.Popen(cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT,
                                            text=True,
                                            encoding='utf-8',
                                            errors='replace',
                                            bufsize=1,
                                            creationflags=creation_flags
                                            )
        except FileNotFoundError:
            self.output_text.insert(tk.END, f"Error: Wrapper script '{wrapper_script}' not found or command error.\n")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            if self.log_file: self.log_file.close(); self.log_file = None
            return
        except Exception as e:
            self.output_text.insert(tk.END, f"Error starting subprocess: {e}\n")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            if self.log_file: self.log_file.close(); self.log_file = None
            return

        # Start the thread to read output
        self.process_thread = threading.Thread(target=self.read_process_output)
        self.process_thread.daemon = True
        self.process_thread.start()

    def read_process_output(self):
        # --- read_process_output method --- (Same as before)
        """Reads output, filters using self._filter_pattern, puts valid lines in queue/log."""
        try:
            if self.process and self.process.stdout:
                for line in iter(self.process.stdout.readline, ''):
                    if line:
                        # Apply the filter
                        if not self._filter_pattern.match(line):
                            # Line does NOT match filter: Process it
                            self.output_queue.put(line)
                            if self.log_file:
                                try:
                                    self.log_file.write(line)
                                except Exception as e:
                                    error_msg = f"--- Error writing to log file: {e} ---\n"
                                    self.output_queue.put(error_msg)
                                    try: self.log_file.close()
                                    except Exception: pass
                                    self.log_file = None # Stop trying
                        # Else: Line matches filter, ignore it
                    else:
                        break # EOF
            else:
                 self.output_queue.put("--- Process or stdout stream not available ---\n")
        except Exception as e:
            if isinstance(e, ValueError) and "I/O operation on closed file" in str(e):
                pass # Expected error after process termination, ignore
            else:
                self.output_queue.put(f"--- Error reading process output: {e} ---\n")
        finally:
            # Cleanup
            process_return_code = None
            if self.process:
                if self.process.stdout and not self.process.stdout.closed:
                     try: self.process.stdout.close()
                     except Exception: pass
                if self.process.poll() is None:
                    try:
                        process_return_code = self.process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        process_return_code = self.process.poll()
                    except Exception:
                        process_return_code = self.process.poll()
                else:
                    process_return_code = self.process.poll()

            if self.log_file:
                try:
                    self.log_file.write(f"\n--- Process ended with return code: {process_return_code} ---\n")
                    self.log_file.close()
                except Exception as e:
                    self.output_queue.put(f"--- Error closing log file: {e} ---\n")
                finally:
                     self.log_file = None

            self.output_queue.put(f"Process finished/terminated with return code: {process_return_code}\n")
            self.output_queue.put("__PROCESS_DONE__")

    def poll_output_queue(self):
        # --- poll_output_queue method --- (Same as before)
        """Checks queue for new output and updates the GUI."""
        process_done_received = False
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line == "__PROCESS_DONE__":
                    process_done_received = True
                else:
                    self.output_text.insert(tk.END, line)
                    self.output_text.see(tk.END)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error polling output queue: {e}")

        if process_done_received:
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.process = None
            self.process_thread = None
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break

        self.after(100, self.poll_output_queue)

    def stop_process(self):
        # --- stop_process method --- (Same as before)
        """Attempts to gracefully terminate the process, then forcefully kills it if necessary."""
        if not self.process or self.process.poll() is not None:
            self.output_text.insert(tk.END, "Process not running or already stopped.\n")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            return

        pid = self.process.pid
        self.output_text.insert(tk.END, f"Attempting to stop process with PID: {pid}...\n")
        self.output_text.see(tk.END)
        self.stop_button.config(state="disabled")

        terminated_gracefully = False
        killed = False

        if sys.platform.startswith("win"):
            # Windows: Use taskkill
            self.output_text.insert(tk.END, "Using taskkill /F /T on Windows...\n")
            try:
                result = subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                    check=True, capture_output=True, text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                self.output_text.insert(tk.END, f"taskkill successful: {result.stdout}\n")
                killed = True
            except subprocess.CalledProcessError as e:
                if "not found" in e.stderr.lower():
                     self.output_text.insert(tk.END, f"Process PID {pid} not found (already terminated?).\n")
                     killed = True
                else:
                    self.output_text.insert(tk.END, f"taskkill failed: {e.stderr}\n")
            except FileNotFoundError:
                 self.output_text.insert(tk.END, "Error: taskkill command not found.\n")
            except Exception as e:
                self.output_text.insert(tk.END, f"Error running taskkill: {e}\n")
        else:
            # Linux/Mac: Try terminate, wait, then kill
            self.output_text.insert(tk.END, "Sending terminate signal...\n")
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                    self.output_text.insert(tk.END, "Process terminated gracefully.\n")
                    terminated_gracefully = True
                except subprocess.TimeoutExpired:
                    self.output_text.insert(tk.END, "Process did not terminate gracefully, sending kill signal...\n")
                    try:
                        self.process.kill()
                        try:
                            self.process.wait(timeout=1)
                            self.output_text.insert(tk.END, "Process killed.\n")
                        except subprocess.TimeoutExpired:
                            self.output_text.insert(tk.END, "Process kill signal sent, but wait timed out.\n")
                        except Exception as wait_err:
                             self.output_text.insert(tk.END, f"Error waiting after kill: {wait_err}\n")
                        killed = True
                    except Exception as kill_err:
                        self.output_text.insert(tk.END, f"Error sending kill signal: {kill_err}\n")
                except Exception as wait_err:
                     self.output_text.insert(tk.END, f"Error waiting for termination: {wait_err}\n")
            except Exception as term_err:
                self.output_text.insert(tk.END, f"Error sending terminate signal (process might have already exited): {term_err}\n")
                if self.process.poll() is not None:
                    terminated_gracefully = True

        # Final Cleanup handled by read_process_output finally block

    def on_closing(self):
        # --- on_closing method --- (Same as before)
        """Handle window closing event."""
        self.output_text.insert(tk.END, "Window closing, attempting to stop process...\n")
        self.stop_process()
        self.destroy()

if __name__ == "__main__":
    app = RunProcessApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

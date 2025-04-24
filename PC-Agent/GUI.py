import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading
import sys
import queue  # 用来在线程间传递输出数据
import time # Import time for timestamping log file

class RunProcessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PC Agent GUI")
        self.geometry("900x700")

        # 定义各项参数对应的变量（此处省略部分参数）
        self.instruction = tk.StringVar(value="Using Edge, add a adidas hat under $20 to cart in amazon.")
        self.icon_caption = tk.BooleanVar(value=False)
        self.use_som = tk.BooleanVar(value=True)
        self.draw_text_box = tk.BooleanVar(value=False)
        self.pc_type = tk.StringVar(value="windows")
        self.enable_reflection = tk.BooleanVar(value=True) # Renamed from disable_reflection for clarity
        self.enable_memory = tk.BooleanVar(value=True)     # Renamed from disable_memory
        self.enable_eval = tk.BooleanVar(value=True)       # Renamed from disable_eval
        self.enable_reward = tk.BooleanVar(value=False)    # 新增 reward 开关
        self.model_backend = tk.StringVar(value="gpt")

        self.build_gui()
        self.process = None
        self.process_thread = None
        self.output_queue = queue.Queue()
        self.log_file = None # Add variable to hold the log file object
        # 开始轮询输出队列
        self.poll_output_queue()

    def build_gui(self):
        # --- GUI layout remains the same ---
        # (Copied your layout code here for completeness)
        # 第一行：Instruction 输入框
        ttk.Label(self, text="Instruction:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self, textvariable=self.instruction, width=80).grid(row=0, column=1, columnspan=4, sticky="ew", padx=5, pady=5) # Adjusted columnspan and sticky
        self.grid_columnconfigure(1, weight=1) # Allow column 1 to expand

        # 第二行：功能选项
        ttk.Checkbutton(self, text="Enable Icon Caption", variable=self.icon_caption).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable SOM", variable=self.use_som).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable Draw Text Box", variable=self.draw_text_box).grid(row=1, column=2, sticky="w", padx=5, pady=5)

        # 第三行：系统类型选择 和 Model selection
        ttk.Label(self, text="PC Type:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        pc_frame = ttk.Frame(self)
        pc_frame.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(pc_frame, text="Windows", variable=self.pc_type, value="windows").pack(side=tk.LEFT)
        ttk.Radiobutton(pc_frame, text="Mac", variable=self.pc_type, value="mac").pack(side=tk.LEFT, padx=10)

        ttk.Label(self, text="Model:").grid(row=2, column=2, sticky="e", padx=5, pady=5) # Align right
        ttk.Combobox(self, textvariable=self.model_backend, values=["gpt", "gemini"], state="readonly", width=10).grid(row=2, column=3, sticky="w", padx=5, pady=5)


        # 第四行：功能开关
        ttk.Checkbutton(self, text="Action Scaling", variable=self.enable_reflection).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable Memory", variable=self.enable_memory).grid(row=3, column=2, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Planning Scaling", variable=self.enable_eval).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Offline Reward", variable=self.enable_reward).grid(row=3, column=3, sticky="w", padx=5, pady=5)

        # 第五行：Start 和 Stop 按钮
        button_frame = ttk.Frame(self)
        button_frame.grid(row=4, column=0, columnspan=4, pady=10) # Center buttons
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_process)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_process, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 第六行：输出日志的滚动文本框
        ttk.Label(self, text="Process Output:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.output_text = ScrolledText(self, wrap="word", width=100, height=30)
        self.output_text.grid(row=6, column=0, columnspan=5, sticky="nsew", padx=5, pady=5) # Use columnspan 5
        self.grid_rowconfigure(6, weight=1) # Allow text area row to expand


    def start_process(self):
        if self.process and self.process.poll() is None:
            self.output_text.insert(tk.END, "Process already running.\n")
            return

        if sys.platform.startswith("win"):
            wrapper_script = "run_with_env.bat"
            # Use 'call' in cmd to ensure environment variables from bat might be set correctly
            # for the subsequent command, though Popen usually handles this.
            # More importantly, using cmd /c directly executes the bat file.
            base_cmd = ["cmd", "/c", wrapper_script]
        else:
            wrapper_script = "run_with_env.sh"
            base_cmd = ["bash", wrapper_script]

        # Arguments passed to the wrapper script (which then passes them to run.py)
        run_py_args = [
            "--instruction", self.instruction.get(),
            "--icon_caption", "1" if self.icon_caption.get() else "0",
            "--use_som", "1" if self.use_som.get() else "0",
            "--draw_text_box", "1" if self.draw_text_box.get() else "0",
            "--pc_type", self.pc_type.get(),
            "--direct_print" # Assuming this tells run.py to print directly
        ]

        # Add conditional arguments for run.py
        if not self.enable_reflection.get():
            run_py_args.append("--disable_reflection")
        if not self.enable_memory.get():
            run_py_args.append("--disable_memory")
        if not self.enable_eval.get():
            run_py_args.append("--disable_eval")
        # Only add --enable_reward if --enable_eval is also active (as per original logic)
        if self.enable_eval.get() and self.enable_reward.get():
             run_py_args.append("--enable_reward")

        # Add backend model selection for run.py
        run_py_args.extend(["--model_backend", self.model_backend.get()])

        # Combine base command with arguments for the wrapper script
        cmd = base_cmd + run_py_args

        # --- File Logging Setup ---
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"process_output_{timestamp}.log"
        try:
            # Open the log file in write mode with UTF-8 encoding and line buffering
            self.log_file = open(log_filename, "w", encoding='utf-8', buffering=1)
        except IOError as e:
             self.output_text.insert(tk.END, f"Error opening log file {log_filename}: {e}\n")
             self.log_file = None # Ensure log_file is None if opening failed
             # Optionally disable start button again or handle error differently
             return # Stop if we can't open the log file

        # --- Start Process ---
        self.output_text.insert(tk.END, f"Saving output also to: {log_filename}\n")
        self.output_text.insert(tk.END, "Starting process with command:\n" + " ".join(cmd) + "\n")
        self.output_text.see(tk.END)
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        try:
            self.process = subprocess.Popen(cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT,
                                            text=True, # Decode output as text
                                            encoding='utf-8', # Specify encoding
                                            errors='replace', # Handle potential decoding errors
                                            bufsize=1) # Line buffered
        except FileNotFoundError:
             self.output_text.insert(tk.END, f"Error: Wrapper script '{wrapper_script}' not found. Make sure it's in the correct path.\n")
             self.start_button.config(state="normal")
             self.stop_button.config(state="disabled")
             if self.log_file:
                 self.log_file.close()
                 self.log_file = None
             return
        except Exception as e:
             self.output_text.insert(tk.END, f"Error starting subprocess: {e}\n")
             self.start_button.config(state="normal")
             self.stop_button.config(state="disabled")
             if self.log_file:
                 self.log_file.close()
                 self.log_file = None
             return


        # Start the thread to read output
        self.process_thread = threading.Thread(target=self.read_process_output)
        self.process_thread.daemon = True
        self.process_thread.start()

    def read_process_output(self):
        """Reads output from the process pipe, puts it in the queue, and writes to log file."""
        try:
            # Ensure stdout is available before reading
            if self.process and self.process.stdout:
                 for line in iter(self.process.stdout.readline, ''): # Read line by line until EOF
                    if line:
                        # Put line in queue for GUI update
                        self.output_queue.put(line)
                        # Write line to the log file if it's open
                        if self.log_file:
                            try:
                                self.log_file.write(line)
                                # self.log_file.flush() # Not strictly needed with buffering=1
                            except Exception as e:
                                # Log error to GUI, maybe stop writing to file?
                                self.output_queue.put(f"--- Error writing to log file: {e} ---\n")
                                # Consider closing self.log_file here if errors persist
                    else:
                         # Should not happen with iter(..., '') unless process closes stream unexpectedly
                         break
            else:
                # Handle case where process or stdout is None unexpectedly
                 self.output_queue.put("--- Process or stdout stream not available for reading ---\n")

        except Exception as e:
            # Catch potential errors during reading (e.g., if process terminates abruptly)
            self.output_queue.put(f"--- Error reading process output: {e} ---\n")
        finally:
            # --- Cleanup after reading finishes or errors occur ---
            if self.process:
                # Ensure process resources are cleaned up if readline loop finished
                if self.process.stdout:
                    self.process.stdout.close()
                # Wait for the process to truly terminate
                self.process.wait()

            # Close the log file if it's open
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception as e:
                    # Log error during file closing if necessary
                     self.output_queue.put(f"--- Error closing log file: {e} ---\n")
                self.log_file = None # Set to None after closing

            # Signal GUI that the process is done
            self.output_queue.put("Process finished.\n")
            self.output_queue.put("__PROCESS_DONE__")


    def poll_output_queue(self):
        """Checks the queue for new output and updates the GUI."""
        try:
            while True: # Process all available lines in the queue
                line = self.output_queue.get_nowait()
                if line == "__PROCESS_DONE__":
                    # Process finished, reset buttons
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.process = None # Clear the process variable
                    # If the process terminated itself, ensure thread is joined (optional but good practice)
                    # if self.process_thread and self.process_thread.is_alive():
                    #    self.process_thread.join(timeout=1.0) # Give thread time to finish cleanup
                    self.process_thread = None
                else:
                    self.output_text.insert(tk.END, line)
                    self.output_text.see(tk.END) # Scroll to the end
        except queue.Empty:
            # Queue is empty, do nothing this cycle
            pass
        except Exception as e:
            # Handle potential errors during GUI update (less likely)
            print(f"Error polling output queue: {e}") # Print to console as GUI might be affected

        # Schedule the next check
        self.after(100, self.poll_output_queue)


    def stop_process(self):
        """Terminates the running process and cleans up."""
        if self.process and self.process.poll() is None: # Check if process exists and is running
            self.output_text.insert(tk.END, "Terminating process...\n")
            self.output_text.see(tk.END)
            try:
                self.process.terminate() # Send SIGTERM (or equivalent)
                # Optionally wait a short time and then force kill if needed
                # self.process.wait(timeout=2)
            except Exception as e:
                 self.output_text.insert(tk.END, f"Error terminating process: {e}\n")

            # The __PROCESS_DONE__ signal will eventually be put by read_process_output
            # after the process truly exits and the thread finishes cleanup.
            # However, we disable the stop button immediately for responsiveness.
            self.stop_button.config(state="disabled")
            # We might re-enable start button here, but it's safer to wait for __PROCESS_DONE__
            # self.start_button.config(state="normal")
        else:
            self.output_text.insert(tk.END, "Process not running or already stopped.\n")
            # Ensure buttons are in correct state if stop is called when not running
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")

        # Force close log file just in case thread hasn't finished yet
        if self.log_file:
            try:
                self.log_file.close()
            except Exception as e:
                 self.output_text.insert(tk.END, f"--- Error closing log file during stop: {e} ---\n")
            self.log_file = None


    def on_closing(self):
        """Handle window closing event."""
        self.stop_process() # Attempt to stop any running process
        self.destroy() # Close the Tkinter window

if __name__ == "__main__":
    app = RunProcessApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing) # Handle window close button
    app.mainloop()
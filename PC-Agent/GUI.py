import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading
import sys
import queue  # 用来在线程间传递输出数据

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
        self.enable_reflection = tk.BooleanVar(value=True)
        self.enable_memory = tk.BooleanVar(value=True)
        self.enable_eval = tk.BooleanVar(value=True)
        self.model_backend = tk.StringVar(value="gpt")
        
        self.build_gui()
        self.process = None
        self.process_thread = None
        self.output_queue = queue.Queue()
        # 开始轮询输出队列
        self.poll_output_queue()

    def build_gui(self):
        # 第一行：Instruction 输入框
        ttk.Label(self, text="Instruction:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self, textvariable=self.instruction, width=80).grid(row=0, column=1, columnspan=3, sticky="w", padx=5, pady=5)
        
        # 第二行：功能选项
        ttk.Checkbutton(self, text="Enable Icon Caption", variable=self.icon_caption).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable SOM", variable=self.use_som).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable Draw Text Box", variable=self.draw_text_box).grid(row=1, column=2, sticky="w", padx=5, pady=5)
        
        # 第三行：系统类型选择
        ttk.Label(self, text="PC Type:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(self, text="Windows", variable=self.pc_type, value="windows").grid(row=2, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(self, text="Mac", variable=self.pc_type, value="mac").grid(row=2, column=2, sticky="w", padx=5, pady=5)
        # Model selection
        ttk.Label(self, text="Model:").grid(row=2, column=3, sticky="w", padx=5, pady=5)
        ttk.Combobox(self, textvariable=self.model_backend, values=["gpt", "gemini"], state="readonly", width=10).grid(row=2, column=4, sticky="w", padx=5, pady=5)
        
        # 第四行：功能开关
        ttk.Checkbutton(self, text="Enable Reflection", variable=self.enable_reflection).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable Memory", variable=self.enable_memory).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(self, text="Enable Eval", variable=self.enable_eval).grid(row=3, column=2, sticky="w", padx=5, pady=5)
        
        # 第五行：Start 和 Stop 按钮
        self.start_button = ttk.Button(self, text="Start", command=self.start_process)
        self.start_button.grid(row=4, column=0, padx=5, pady=10)
        self.stop_button = ttk.Button(self, text="Stop", command=self.stop_process, state="disabled")
        self.stop_button.grid(row=4, column=1, padx=5, pady=10)
        
        # 第六行：输出日志的滚动文本框
        ttk.Label(self, text="Process Output:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.output_text = ScrolledText(self, wrap="word", width=100, height=30)
        self.output_text.grid(row=6, column=0, columnspan=4, padx=5, pady=5)

    def start_process(self):
        if sys.platform.startswith("win"):
            wrapper_script = "run_with_env.bat"
            cmd = [ "cmd", "/c", wrapper_script,
                    "--instruction", self.instruction.get(),
                    "--icon_caption", "1" if self.icon_caption.get() else "0",
                    "--use_som", "1" if self.use_som.get() else "0",
                    "--draw_text_box", "1" if self.draw_text_box.get() else "0",
                    "--pc_type", self.pc_type.get(),
                    "--direct_print"]
        else:
            wrapper_script = "run_with_env.sh"
            cmd = [ "bash", wrapper_script,
                   "--instruction", self.instruction.get(),
                   "--icon_caption", "1" if self.icon_caption.get() else "0",
                   "--use_som", "1" if self.use_som.get() else "0",
                   "--draw_text_box", "1" if self.draw_text_box.get() else "0",
                   "--pc_type", self.pc_type.get(),
                   "--direct_print"]
            
        # 构造传递给 run.py 的命令行参数，注意加上 -u 参数启动无缓冲
        
        if not self.enable_reflection.get():
            cmd.append("--disable_reflection")
        if not self.enable_memory.get():
            cmd.append("--disable_memory")
        if not self.enable_eval.get():
            cmd.append("--disable_eval")
        # Add backend model selection
        cmd.extend(["--model_backend", self.model_backend.get()])
        
        self.output_text.insert(tk.END, "Starting process with command:\n" + " ".join(cmd) + "\n")
        self.output_text.see(tk.END)
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        self.process = subprocess.Popen(cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        text=True,
                                        bufsize=1)
        self.process_thread = threading.Thread(target=self.read_process_output)
        self.process_thread.daemon = True
        self.process_thread.start()

    def read_process_output(self):
        while True:
            line = self.process.stdout.readline()
            if line:
                # 把输出行放入队列，让主线程更新 GUI
                self.output_queue.put(line)
            else:
                break
        self.process.stdout.close()
        self.process.wait()
        self.output_queue.put("Process finished.\n")
        # 在子进程结束后也恢复按钮状态
        self.output_queue.put("__PROCESS_DONE__")

    def poll_output_queue(self):
        # 每隔一小段时间检查队列是否有新数据
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line == "__PROCESS_DONE__":
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                else:
                    self.output_text.insert(tk.END, line)
                    self.output_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self.poll_output_queue)

    def stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.output_text.insert(tk.END, "Process terminated by user.\n")
            self.output_text.see(tk.END)
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

if __name__ == "__main__":
    app = RunProcessApp()
    app.mainloop()

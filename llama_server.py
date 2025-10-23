import signal
from PyQt6.QtCore import (
    QThread
)
import subprocess
import atexit
from pathlib import Path
import psutil

class Llama_cpp(QThread):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self._is_running = True

    def run(self):
        self._is_running = True
        llama_path = Path("llama") / "llama-server"
        command = [str(llama_path), "-m", self.options['model_path'], "--host", "127.0.0.1", "--port", str(self.options['port']), "-n", "-1"]
        if self.options['threads'] > 0:
            command += ["-t", str(self.options['threads'])]
        if self.options['gpu_layers'] > 0:
            command += ["--n-gpu-layers", str(self.options['gpu_layers'])]
        if self.options['batch_size'] > 0:
            command += ["-b", str(self.options['batch_size'])]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, encoding="utf-8")

        atexit.register(self.process.terminate)

    def stop(self):
        self._is_running = False
        try:
            try:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=1)
                return
            except subprocess.TimeoutExpired:
                pass
            
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
                return
            except subprocess.TimeoutExpired:
                pass

            self.process.kill()
            self.process.wait(timeout=1)
        except Exception as e:
            print(f"Error terminating llama.cpp server: {e}")

def _matches(proc: psutil.Process) -> bool:
    try:
        name = (proc.info.get("name") or "").lower()
        if "llama-server" in name:
            return True
        cmd = " ".join(proc.info.get("cmdline") or []).lower()
        return "llama-server" in cmd
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False

def is_llama_server_running() -> bool:
    return any(_matches(p) for p in psutil.process_iter(attrs=["name", "cmdline"]))

def kill_llama_server(timeout: float = 3.0) -> None:
    procs = [p for p in psutil.process_iter(attrs=["name", "cmdline"]) if _matches(p)]
    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
import os
import time
from datetime import datetime
import psutil
import subprocess

class TrainingMonitor:
    
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
    
    def get_gpu_vram_from_nvidia_smi(self):
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
            used, total = output.strip().split(',')
            return float(used), float(total)
        except Exception:
            return None, None
    
    def info(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        
        host_mem = self.process.memory_info().rss / (1024**2)
        used, total = self.get_gpu_vram_from_nvidia_smi()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if used is not None:
            print(f"      -{timestamp} END | 耗時：{int(mins)}分{int(secs)}秒 | Host RAM: {host_mem:.1f}MB; GPU VRAM: {used:.1f} MB / {total:.1f} MB")
        else:
            print(f"      -{timestamp} END | 耗時：{int(mins)}分{int(secs)}秒 | Host RAM: {host_mem:.1f}MB; GPU 偵測不到或無 TF 支援")


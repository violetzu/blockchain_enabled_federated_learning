import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# if tf.config.list_physical_devices('GPU'):


import subprocess

def get_gpu_vram_from_nvidia_smi():
    try:
        # 回傳格式類似： " 123 MiB,  8192 MiB\n"
        output = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        # 拆成 used, total （MB）
        used, total = output.strip().split(',')
        return float(used), float(total)
    except Exception as e:
        return None, None

used, total = get_gpu_vram_from_nvidia_smi()
if used is not None:
    print(f"GPU VRAM：{used:.1f} MB / {total:.1f} MB")
else:
    print("nvidia-smi 無法讀取或未安裝")

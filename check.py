import os, platform, torch, subprocess

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
    print("nvidia-smi OK\n", "\n".join(out.splitlines()[:10]))
except Exception as e:
    print("nvidia-smi ERROR ->", e)

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

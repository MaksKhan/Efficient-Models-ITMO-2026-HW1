"""Run the full fixed-seed grid on one CUDA GPU. Resume by rerunning this file."""
import csv
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import torch
from models import SmallCNN
from equations import memory as predicted_memory

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
CSV_PATH = RESULTS / 'measurements.csv'
BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RNG = np.random.default_rng(20260924)
# Sample from the upper part of the allowed ranges to stress larger workloads.
EXTRA_S = sorted(RNG.choice([s for s in range(128, 513, 16) if s not in BASE_S], 4, replace=False).tolist())
EXTRA_B = sorted(RNG.choice([b for b in range(96, 257) if b & (b - 1)], 3, replace=False).tolist())
SIZES = sorted(BASE_S + EXTRA_S)
BATCHES = sorted(BASE_B + EXTRA_B)
WARMUP_REPEATS = 10
ENERGY_TARGET_SECONDS = 3.0
ENERGY_MIN_REPEATS = 10
ENERGY_MAX_REPEATS = 10000
FIELDS = ['image_size','batch','status','latency_s','memory_bytes','energy_j','mean_power_w',
          'predicted_memory_bytes','is_validation','warmup_repeats','latency_repeats',
          'energy_repeats','energy_duration_s','power_samples','power_unique_values','error']


def nvml_reader():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return lambda: float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000
    except Exception:
        try:
            import ctypes
            lib = ctypes.CDLL('libnvidia-ml.so.1')
            assert lib.nvmlInit_v2() == 0
            handle = ctypes.c_void_p()
            assert lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0
            def read():
                milliwatts = ctypes.c_uint()
                rc = lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(milliwatts))
                return float(milliwatts.value) / 1000 if rc == 0 else math.nan
            return read
        except Exception:
            return None


def sample_power(read, stop, samples):
    while not stop.is_set():
        try:
            value = read()
            if math.isfinite(value):
                samples.append((time.perf_counter(), value))
        except Exception:
            pass
        stop.wait(0.03)


def validate_resume(csv_path, hardware_path, hardware):
    if not csv_path.exists():
        return
    if not hardware_path.exists():
        raise RuntimeError('Cannot resume measurements without results/hardware.json')
    if json.loads(hardware_path.read_text()) != hardware:
        raise RuntimeError('Measurement protocol or GPU changed; archive results before rerunning')
    with csv_path.open() as f:
        if csv.DictReader(f).fieldnames != FIELDS:
            raise RuntimeError('Measurement CSV schema changed; archive results before rerunning')


def one_point(model, s, b, read_power):
    row = {'image_size':s,'batch':b,'status':'ok','predicted_memory_bytes':float(predicted_memory(s,b)),
           'latency_s':math.nan,'memory_bytes':math.nan,'energy_j':math.nan,'mean_power_w':math.nan,
           'warmup_repeats':WARMUP_REPEATS,'latency_repeats':0,
           'energy_repeats':0,'energy_duration_s':math.nan,
           'power_samples':0,'power_unique_values':0,'error':''}
    x = None
    try:
        x = torch.randn((b,3,s,s), device='cuda', dtype=torch.float32)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(WARMUP_REPEATS):
                y = model(x)
            torch.cuda.synchronize()
            row['memory_bytes'] = float(torch.cuda.max_memory_allocated())
            del y
            timings = []
            for _ in range(7):
                start = time.perf_counter()
                y = model(x)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
            row['latency_s'] = float(np.median(timings))
            row['latency_repeats'] = len(timings)
            del y
            if read_power is not None:
                # A few seconds are needed to observe several NVML power updates,
                # especially when a single forward takes less than a millisecond.
                n = min(ENERGY_MAX_REPEATS,
                        max(ENERGY_MIN_REPEATS,
                            math.ceil(ENERGY_TARGET_SECONDS / max(row['latency_s'],1e-4))))
                samples = []
                stop = threading.Event()
                thread = threading.Thread(target=sample_power,args=(read_power,stop,samples),daemon=True)
                torch.cuda.synchronize()
                thread.start()
                start = time.perf_counter()
                for _ in range(n):
                    y = model(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                elapsed = end - start
                stop.set(); thread.join(timeout=1)
                active_power = [p for ts,p in samples if start <= ts <= end]
                if active_power:
                    row['mean_power_w'] = float(np.mean(active_power))
                    row['energy_j'] = row['mean_power_w'] * elapsed / n
                    row['power_samples'] = len(active_power)
                    row['power_unique_values'] = len(set(active_power))
                row['energy_repeats'] = n
                row['energy_duration_s'] = elapsed
                del y
    except torch.cuda.OutOfMemoryError as exc:
        row['status'] = 'OOM'
        row['error'] = str(exc).splitlines()[0][:160]
    finally:
        if x is not None: del x
        torch.cuda.empty_cache()
    return row


def main():
    assert torch.cuda.is_available(), 'A CUDA GPU is required for measurements'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260924)
    model = SmallCNN().cuda().eval()
    read_power = nvml_reader()
    device = torch.cuda.get_device_properties(0)
    hardware_path = RESULTS/'hardware.json'
    hardware = {
        'gpu':device.name,'gpu_total_memory_bytes':device.total_memory,
        'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'power_sensor_available':read_power is not None,
        'seed':20260924,'image_sizes':SIZES,'batch_sizes':BATCHES,
        'benchmark':False,'cudnn_tf32':False,'matmul_tf32':False,
        'warmup_repeats_per_configuration':WARMUP_REPEATS,
        'energy_target_seconds':ENERGY_TARGET_SECONDS,
        'energy_min_repeats':ENERGY_MIN_REPEATS,
        'energy_max_repeats':ENERGY_MAX_REPEATS}
    validate_resume(CSV_PATH, hardware_path, hardware)
    hardware_path.write_text(json.dumps(hardware,indent=2))
    print('GPU:',device.name,'power sensor:',read_power is not None,flush=True)
    done = set()
    csv_exists = CSV_PATH.exists()
    if csv_exists:
        with CSV_PATH.open() as f:
            for row in csv.DictReader(f): done.add((int(row['image_size']),int(row['batch'])))
    with CSV_PATH.open('a',newline='') as f:
        writer = csv.DictWriter(f,FIELDS)
        if not csv_exists: writer.writeheader()
        # Interleaved holdout covering both dimensions, fixed before seeing measurements.
        for s in SIZES:
            for b in BATCHES:
                if (s,b) in done: continue
                row = one_point(model,s,b,read_power)
                row['is_validation'] = int((SIZES.index(s)*12+BATCHES.index(b)) % 5 == 0)
                writer.writerow(row); f.flush()
                print(f'{s=} {b=} {row["status"]} latency={row["latency_s"]:.5g} energy={row["energy_j"]:.5g}',flush=True)

if __name__ == '__main__': main()

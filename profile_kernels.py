"""Capture CUDA kernel names for representative feasible grid configurations."""
import csv
from pathlib import Path
import torch
from models import SmallCNN

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'kernels.csv'
POINTS=[(32,1),(224,16),(512,16)]


def main():
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    model=SmallCNN().cuda().eval()
    with OUT.open('w',newline='') as f:
        writer=csv.DictWriter(f,['image_size','batch','layer','kernel_name','count'])
        writer.writeheader()
        for s,b in POINTS:
            try:
                x=torch.randn(b,3,s,s,device='cuda')
                with torch.inference_mode():
                    for prefix,sequence in [('features',model.features),('head',model.head)]:
                        for i,layer in enumerate(sequence):
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                                y=layer(x)
                                torch.cuda.synchronize()
                            names={}
                            for event in prof.events():
                                if event.device_type==torch.autograd.DeviceType.CUDA:
                                    names[event.name]=names.get(event.name,0)+1
                            for name,count in sorted(names.items()):
                                writer.writerow({'image_size':s,'batch':b,'layer':f'{prefix}.{i}:{type(layer).__name__}',
                                                 'kernel_name':name,'count':count})
                            x=y
                f.flush()
                print(f'profiled {s=} {b=}',flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f'profile OOM {s=} {b=}',flush=True)

if __name__=='__main__': main()

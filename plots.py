"""Overlay measured grid points with analytical predictions."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from equations import memory, latency, energy

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'; FIG=RESULTS/'figures'; FIG.mkdir(exist_ok=True)


def main():
    with (RESULTS/'measurements.csv').open() as f: rows=list(csv.DictReader(f))
    summary=json.loads((RESULTS/'theta.json').read_text()); theta=summary['theta']
    # Coefficients were fitted elsewhere; plot measurements only from the held-out split.
    held_out=[r for r in rows if int(r['is_validation'])==1]
    ok=[r for r in held_out if r['status']=='ok']
    sizes=sorted({int(r['image_size']) for r in rows}); batches=sorted({int(r['batch']) for r in rows})
    sb=np.array([[float(r['image_size']),float(r['batch'])] for r in ok]); s,b=sb[:,0],sb[:,1]
    for key,unit,func,scale,file in [
        ('latency_s','ms',lambda x,y: latency(x,y,theta),1e3,'latency.png'),
        ('memory_bytes','MiB',memory,1/2**20,'memory.png'),
        ('energy_j','J',lambda x,y: energy(x,y,theta) if theta['power'] is not None else np.full_like(x,np.nan),1,'energy.png')]:
        actual=np.array([float(r[key]) for r in ok])*scale
        predicted=np.asarray(func(s,b))*scale
        finite=np.isfinite(actual)&np.isfinite(predicted)
        if not finite.any(): continue
        fig,axs=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
        axs[0].scatter(predicted[finite],actual[finite],s=28,alpha=.8,
                       label='held-out measurement',marker='^')
        top=max(np.max(actual[finite]),np.max(predicted[finite])); axs[0].plot([0,top],[0,top],'k--',label='prediction = measurement')
        axs[0].set(xlabel=f'Predicted {unit}',ylabel=f'Measured {unit}',title=f'{key}: held-out (S, B)')
        axs[0].legend(); axs[0].grid(alpha=.25)
        # Full-grid surface shown as predicted curves for selected batches, with measured points.
        colors=plt.cm.viridis(np.linspace(0,1,len(batches)))
        for i,batch in enumerate(batches):
            curve_s=np.arange(min(sizes),max(sizes)+1,16)
            curve=np.asarray(func(curve_s,batch))*scale
            axs[1].plot(curve_s,curve,color=colors[i],lw=1,alpha=.8,label=f'B={batch}')
            subset=finite&(b==batch)
            axs[1].scatter(s[subset],actual[subset],color=colors[i],s=18,marker='o')
        axs[1].set(xlabel='Image size S (pixels)',ylabel=f'{key} ({unit})',title='Lines: prediction; points: held-out measurement')
        axs[1].set_yscale('log'); axs[1].grid(alpha=.25)
        axs[1].legend(ncol=3,fontsize=7)
        fig.savefig(FIG/file,dpi=160); plt.close(fig)

if __name__=='__main__': main()

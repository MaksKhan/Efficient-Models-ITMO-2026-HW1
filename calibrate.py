"""Fit theta on calibration points and report error on held-out points."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from equations import flops, memory, bytes_moved, latency, energy

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'


def metrics(truth,pred):
    truth=np.asarray(truth,dtype=float); pred=np.asarray(pred,dtype=float)
    return {'n':len(truth),'mape_pct':float(100*np.mean(np.abs(pred-truth)/np.maximum(np.abs(truth),1e-12))),
            'median_ape_pct':float(100*np.median(np.abs(pred-truth)/np.maximum(np.abs(truth),1e-12))),
            'rmse':float(np.sqrt(np.mean((pred-truth)**2)))}


def main():
    with (RESULTS/'measurements.csv').open() as f: rows=list(csv.DictReader(f))
    ok=[r for r in rows if r['status']=='ok']
    train=[r for r in ok if int(r['is_validation'])==0]
    val=[r for r in ok if int(r['is_validation'])==1]
    if len(train)<3: raise RuntimeError('Need GPU measurements first')
    def arrays(rr):
        return np.array([float(r['image_size']) for r in rr]),np.array([float(r['batch']) for r in rr])
    s,b=arrays(train); t=np.array([float(r['latency_s']) for r in train])
    # Fit in log space to enforce positive throughput and bandwidth.
    def unpack(x): return {'tau':float(np.exp(x[0])),'compute_rate':float(np.exp(x[1])),
                           'bandwidth':float(np.exp(x[2]))}
    x0=np.log([max(np.min(t)/2,1e-6),1e12,1e11])
    fit=least_squares(lambda x: np.log(latency(s,b,unpack(x)))-np.log(t),x0,
                      bounds=(np.log([1e-8,1e8,1e6]),np.log([10,1e16,1e14])),max_nfev=3000)
    theta=unpack(fit.x)
    power_train=[r for r in train if np.isfinite(float(r['energy_j']))]
    if len(power_train)>=3:
        es,eb=arrays(power_train)
        y=np.array([float(r['energy_j']) for r in power_train])
        features=np.column_stack([latency(es,eb,theta),flops(es,eb),bytes_moved(es,eb)])
        # Unit scaling keeps the nonnegative solve numerically stable.
        feature_scale=np.maximum(features.max(axis=0),1e-30)
        weights=1/np.maximum(y,1e-6)
        solve=lsq_linear((features/feature_scale)*weights[:,None],y*weights,bounds=(0,np.inf),tol=1e-12,max_iter=1000)
        p,ef,em=solve.x/feature_scale
        theta.update(power=float(p),energy_per_flop=float(ef),energy_per_byte=float(em))
    else:
        theta.update(power=None,energy_per_flop=None,energy_per_byte=None)
    summary={'theta':theta,'train_count':len(train),'validation_count':len(val),
             'oom_count':sum(r['status']=='OOM' for r in rows),
             'latency_fit_converged':bool(fit.success),
             'latency_train':metrics(t,latency(s,b,theta)),
             'memory_train':metrics([float(r['memory_bytes']) for r in train],memory(s,b))}
    f_values=np.asarray(flops(s,b)); q_values=np.asarray(bytes_moved(s,b))
    summary['flops_bytes_correlation']=float(np.corrcoef(f_values,q_values)[0,1])
    if theta['power'] is not None:
        energy_train_s,energy_train_b=arrays(power_train)
        energy_truth=np.array([float(r['energy_j']) for r in power_train])
        design=np.column_stack([latency(energy_train_s,energy_train_b,theta),
                                flops(energy_train_s,energy_train_b),
                                bytes_moved(energy_train_s,energy_train_b)])
        design=design/np.maximum(design.max(axis=0),1e-30)
        summary['energy_design_condition_number']=float(np.linalg.cond(design/energy_truth[:,None]))
        if 'power_unique_values' in power_train[0]:
            summary['power_unique_values_min']=min(int(r['power_unique_values']) for r in power_train)
            summary['energy_duration_min_s']=min(float(r['energy_duration_s']) for r in power_train)
    if val:
        vs,vb=arrays(val)
        summary['latency_validation']=metrics([float(r['latency_s']) for r in val],latency(vs,vb,theta))
        summary['memory_validation']=metrics([float(r['memory_bytes']) for r in val],memory(vs,vb))
    if theta['power'] is not None:
        for name,subset in [('train',train),('validation',val)]:
            subset=[r for r in subset if np.isfinite(float(r['energy_j']))]
            if subset:
                ss,bb=arrays(subset)
                summary[f'energy_{name}']=metrics([float(r['energy_j']) for r in subset],energy(ss,bb,theta))
    (RESULTS/'theta.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__': main()

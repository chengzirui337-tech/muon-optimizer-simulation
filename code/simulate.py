"""Reproduce every experiment in the Muon course report; no downloaded data.

Python 3.12; numpy and matplotlib. Run from any directory:
    python simulate.py
Raw records, selected hyperparameters and figures are written beside this file.
The toy predictor is an MLP on synthetic token contexts, not a pretrained LLM.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT, FIG = ROOT / "results", ROOT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "legend.frameon": False})
COLORS = {"AdamW": "#53657A", "Muon-NS5": "#B15832", "Muon-SVD": "#237E83"}

def polar(m):
    """Canonical partial polar factor: null singular values stay zero."""
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    keep = s > (max(m.shape) * np.finfo(m.dtype).eps * (s[0] if s.size else 0))
    return (u[:, keep] @ vt[keep]) if keep.any() else np.zeros_like(m)

def ns(m, steps=5, quintic=True):
    norm = np.linalg.norm(m)
    if norm == 0:
        return np.zeros_like(m)
    x = m / norm
    transposed = x.shape[0] > x.shape[1]
    if transposed: x = x.T
    a, b, c = (3.4445, -4.7750, 2.0315) if quintic else (1.5, -0.5, 0.)
    for _ in range(steps):
        aat = x @ x.T
        x = a * x + (b * aat + c * (aat @ aat)) @ x
    return x.T if transposed else x

def savefig(fig, name):
    fig.savefig(FIG / (name + ".pdf"))
    fig.savefig(FIG / (name + ".png"), dpi=190)
    plt.close(fig)

def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def experiment_quadratic():
    rng = np.random.default_rng(7)
    m, n, L = 24, 12, 10.
    h = np.geomspace(1., L, m)[:, None]
    target = rng.normal(size=(m,n)) / np.sqrt(m)
    delta = float(.5 * np.sum(h * target**2))
    rows = []
    for horizon in [100, 400, 1600, 6400]:
        eta = np.sqrt(2 * delta / (L * n * horizon))
        w = np.zeros_like(target); gradients = []; max_violation = -np.inf
        for t in range(horizon):
            g = h * (w - target)
            nuc = np.linalg.svd(g, compute_uv=False).sum()
            gradients.append(nuc)
            f = .5 * np.sum(h * (w - target)**2)
            wnext = w - eta * polar(g)
            fnext = .5 * np.sum(h * (wnext - target)**2)
            max_violation = max(max_violation, fnext - (f - eta*nuc + L*n*eta**2/2))
            w = wnext
        bound = delta / (eta*horizon) + L*n*eta/2
        rows.append(dict(T=horizon, eta=float(eta), average_nuclear_gradient=float(np.mean(gradients)),
            theorem_bound=float(bound), final_loss=float(fnext), max_descent_violation=float(max_violation)))
        assert np.mean(gradients) <= bound + 1e-9
        assert max_violation <= 1e-9
    write_csv("quadratic.csv", rows)
    return dict(delta=delta, L=L, m=m, n=n, records=rows)

def experiment_ns():
    rng = np.random.default_rng(8)
    u = np.linalg.qr(rng.normal(size=(32,16)))[0]
    v = np.linalg.qr(rng.normal(size=(16,16)))[0]
    s = np.geomspace(1, 1e-4, 16)
    mat = (u*s) @ v.T; p = u @ v.T
    rows=[]
    for k in [1,2,3,5,8,12,20,30]:
        for kind, quintic in [("NS5-polynomial", True), ("Classical-cubic", False)]:
            q = ns(mat, k, quintic)
            vals = np.linalg.svd(q, compute_uv=False)
            rows.append(dict(polynomial=kind, steps=k,
                relative_F_error=float(np.linalg.norm(q-p)/np.linalg.norm(p)),
                operator_error=float(np.linalg.norm(q-p,2)),
                minimum_singular=float(vals[-1]), maximum_singular=float(vals[0])))
    write_csv("orthogonalization.csv", rows)
    return rows

# All 256 two-token contexts form disjoint train/validation/test sets.
# A known stochastic rule provides exact soft targets, so no label sampling
# noise or accidental cross-split duplicate contexts obscure optimization.
V, HIDDEN, STEPS, BATCH = 16, 64, 800, 64
def dataset():
    contexts = np.array([(a,b) for a in range(V) for b in range(V)])
    x = np.eye(V)[contexts].reshape(-1, 2*V)
    target = (contexts[:,0] + 2*contexts[:,1]) % V
    y = np.full((V*V, V), .1/V)
    y[np.arange(V*V),target] += .9
    order = np.random.default_rng(2026).permutation(V*V)
    splits = dict(train=order[:192], validation=order[192:224], test=order[224:])
    return x, y, target, splits

def init(seed):
    rng = np.random.default_rng(seed)
    return dict(w=rng.normal(size=(2*V,HIDDEN))*np.sqrt(2/(2*V)), b=np.zeros(HIDDEN),
        head=rng.normal(size=(HIDDEN,V))/np.sqrt(HIDDEN), bias=np.zeros(V))

def forward(p,x):
    z=x@p['w']+p['b']; h=np.tanh(z)
    logits=h@p['head']+p['bias']; logits-=logits.max(axis=1,keepdims=True)
    logp=logits-np.log(np.exp(logits).sum(axis=1,keepdims=True))
    return h,logp

def metrics(p,x,y,target):
    h,logp=forward(p,x)
    ce=float(-np.mean(np.sum(y*logp,axis=1)))
    acc=float(np.mean(logp.argmax(1)==target))
    return ce,acc

def gradients(p,x,y):
    h,logp=forward(p,x); d=(np.exp(logp)-y)/len(x)
    dz=(d@p['head'].T)*(1-h*h)
    return dict(w=x.T@dz,b=dz.sum(0),head=h.T@d,bias=d.sum(0))

def train(method,lr,seed,record=False):
    x,y,targets,splits=dataset(); p=init(seed)
    m={k:np.zeros_like(v) for k,v in p.items()}; v={k:np.zeros_like(vv) for k,vv in p.items()}
    momentum=np.zeros_like(p['w']); rng=np.random.default_rng(seed+10000)
    history=[]; start=time.perf_counter()
    for step in range(1,STEPS+1):
        ids=rng.choice(splits['train'], BATCH, replace=True)
        g=gradients(p,x[ids],y[ids])
        ratio=min(step/40,1.)*(.1+.9*(1+np.cos(np.pi*max(step-40,0)/(STEPS-40)))/2)
        for key in p:
            # Hidden matrix is the sole Muon parameter; head and biases use
            # exactly the same AdamW learning rate in all three conditions.
            decay=.01 if key in ['w','head'] else 0.
            if key=='w' and method!='AdamW':
                momentum=.95*momentum+.05*g[key]
                # Heavy-ball variant, not Nesterov. Positive momentum scaling
                # is irrelevant for exact P, but not the epsilon in finite NS.
                direction=polar(momentum) if method=='Muon-SVD' else ns(momentum)
                scale=.2*np.sqrt(max(p[key].shape))
                p[key]=(1-lr*ratio*decay)*p[key]-lr*ratio*scale*direction
            else:
                adam_lr=lr if method=='AdamW' else .003
                m[key]=.9*m[key]+.1*g[key]; v[key]=.99*v[key]+.01*g[key]**2
                direction=(m[key]/(1-.9**step))/(np.sqrt(v[key]/(1-.99**step))+1e-8)
                # In the AdamW condition the head rate is also fixed at .003.
                if key!='w': adam_lr=.003
                p[key]=(1-adam_lr*ratio*decay)*p[key]-adam_lr*ratio*direction
        if record and (step==1 or step%20==0):
            tr,ta=metrics(p,x[splits['train']],y[splits['train']],targets[splits['train']])
            va,vacc=metrics(p,x[splits['validation']],y[splits['validation']],targets[splits['validation']])
            history.append(dict(method=method,seed=seed,step=step,train_ce=tr,validation_ce=va,
                train_accuracy=ta,validation_accuracy=vacc,elapsed_seconds=time.perf_counter()-start))
    seconds=time.perf_counter()-start
    result={"method":method,"seed":seed,"lr":lr,"seconds":seconds,"weight_norm":float(np.linalg.norm(p['w']))}
    for split,ids in splits.items():
        ce,acc=metrics(p,x[ids],y[ids],targets[ids]); result[split+'_ce']=ce; result[split+'_accuracy']=acc
    return result,history

def experiment_predictor():
    methods=['AdamW','Muon-NS5','Muon-SVD']; selected={}; tuning=[]
    for method in methods:
        grid=[.001,.003,.01,.03,.1,.3,1.] if method=='AdamW' else [.005/3,.005,.015,.045,.135,.405,1.215]
        for lr in grid:
            rec,_=train(method,lr,seed=0); tuning.append(rec)
        selected[method]=min([r for r in tuning if r['method']==method],key=lambda r:r['validation_ce'])['lr']
    write_csv('tuning.csv',tuning)
    records=[]; histories=[]
    for seed in [11,22,33]:
        for method in methods:
            rec,hist=train(method,selected[method],seed,True)
            records.append(rec); histories.extend(hist)
            print(method, seed, 'train',round(rec['train_ce'],4),'test',round(rec['test_ce'],4),flush=True)
    write_csv('predictor_runs.csv',records); write_csv('predictor_history.csv',histories)
    summary={}
    for method in methods:
        runs=[r for r in records if r['method']==method]
        summary[method]={k:dict(mean=float(np.mean([r[k] for r in runs])),sd=float(np.std([r[k] for r in runs],ddof=1)))
            for k in ['train_ce','validation_ce','test_ce','test_accuracy','seconds','weight_norm']}
    return dict(selected_lr=selected, summary=summary, parameters=sum(a.size for a in init(0).values()),
        irreducible_entropy=float(-(.9+.1/V)*np.log(.9+.1/V)-(V-1)*(.1/V)*np.log(.1/V)))

def sanity_checks():
    # Checks target the mathematical claims and manual derivative, not merely
    # re-executing the optimizer's assignments.
    rng=np.random.default_rng(4); a=rng.normal(size=(8,5)); p=polar(a)
    assert abs(np.sum(a*p)-np.linalg.svd(a,compute_uv=False).sum())<1e-10
    assert np.linalg.norm(p,2)<=1+1e-10
    assert np.all(polar(np.zeros((3,2)))==0)
    x,y,_,sp=dataset(); pars=init(0); ids=sp['train'][:7]; grad=gradients(pars,x[ids],y[ids])
    def loss(): return -np.mean(np.sum(y[ids]*forward(pars,x[ids])[1],axis=1))
    for key,index in [('w',(1,3)),('head',(4,2)),('b',(3,)),('bias',(2,))]:
        value=pars[key][index]; eps=1e-6
        pars[key][index]=value+eps; plus=loss(); pars[key][index]=value-eps; minus=loss(); pars[key][index]=value
        assert abs((plus-minus)/(2*eps)-grad[key][index])<1e-7
    splitsets=[set(sp[k]) for k in ['train','validation','test']]
    assert all(not splitsets[i]&splitsets[j] for i in range(3) for j in range(i))

if __name__=='__main__':
    sanity_checks()
    report=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
        platform=platform.platform(),processor=platform.processor(),precision='float64',threads=1),
        quadratic=experiment_quadratic(),orthogonalization=experiment_ns(),predictor=experiment_predictor())
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    from plot_saved import redraw
    redraw()
    print(json.dumps(report['predictor'],indent=2),flush=True)

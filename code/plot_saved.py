"""Redraw report figures from saved CSVs, without rerunning training."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
COLORS={'AdamW':'#53657A','Muon-NS5':'#B15832','Muon-SVD':'#237E83'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'legend.frameon':False})

def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf')
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=190)
    plt.close(fig)

def finish(ax,ncol=2):
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),ncol=ncol,fontsize=8,borderaxespad=0)
    ax.grid(alpha=.15)

def redraw():
    (ROOT/'figures').mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,3.7),layout='constrained'); rows=read('quadratic.csv')
    ts=[float(r['T']) for r in rows]
    ax.loglog(ts,[float(r['average_nuclear_gradient']) for r in rows],'o-',color=COLORS['Muon-SVD'],label='Measured average nuclear gradient')
    ax.loglog(ts,[float(r['theorem_bound']) for r in rows],'s--',color=COLORS['AdamW'],label='Theorem 1 upper bound')
    ax.set(xlabel='Horizon T (fresh run for each T)',ylabel='Average nuclear gradient / upper bound')
    finish(ax,1); save(fig,'quadratic')
    fig,ax=plt.subplots(figsize=(6,3.7),layout='constrained'); rows=read('orthogonalization.csv')
    for kind,c in [('NS5-polynomial',COLORS['Muon-NS5']),('Classical-cubic',COLORS['Muon-SVD'])]:
        rr=[r for r in rows if r['polynomial']==kind]
        ax.semilogy([float(r['steps']) for r in rr],[float(r['relative_F_error']) for r in rr],'o-',color=c,label=kind)
    ax.set(xlabel='Number of matrix iterations',ylabel=r'Relative error $\|Q-P(M)\|_F/\|P(M)\|_F$')
    finish(ax); save(fig,'orthogonalization')
    histories=read('predictor_history.csv')
    for metric,label,filename in [('train_ce','Training cross-entropy (nats)','predictor_train'),
                                  ('validation_ce','Validation cross-entropy (nats)','predictor_validation')]:
        fig,ax=plt.subplots(figsize=(6,3.7),layout='constrained')
        for method in COLORS:
            curves=np.array([[float(r[metric]) for r in histories if r['method']==method and int(r['seed'])==s] for s in [11,22,33]])
            ts=[float(r['step']) for r in histories if r['method']==method and int(r['seed'])==11]
            mean=curves.mean(0); sd=curves.std(0,ddof=1)
            ax.plot(ts,mean,label=method,color=COLORS[method])
            ax.fill_between(ts,mean-sd,mean+sd,color=COLORS[method],alpha=.12)
        ax.set(xlabel='Training step',ylabel=label); finish(ax,3); save(fig,filename)

if __name__=='__main__': redraw()

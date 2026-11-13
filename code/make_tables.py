"""Convert saved measurements to LaTeX tables; does not run training."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent.parent
data=json.loads((root/'results/summary.json').read_text(encoding='utf-8'))
rows=[]
for r in data['quadratic']['records']:
    rows.append(f"{r['T']} & {r['eta']:.5f} & {r['average_nuclear_gradient']:.4f} & {r['theorem_bound']:.4f} & {r['final_loss']:.6f} \\\\")
quad_header=r'''\begin{tabular}{rrrrr}\toprule
$T$ & $\eta$ & 平均核范数 & 理论上界 & 终点损失\\\midrule
'''
(root/'results/quadratic_table.tex').write_text(quad_header+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n',encoding='utf-8')
rows=[]
for method,r in data['predictor']['summary'].items():
    vals=[]
    for k in ['train_ce','test_ce','test_accuracy','seconds']:
        scale=100 if k=='test_accuracy' else 1
        precision = 4 if k in ['train_ce','seconds'] else (1 if k=='test_accuracy' else 3)
        vals.append(f"${r[k]['mean']*scale:.{precision}f}\\pm {r[k]['sd']*scale:.{precision}f}$")
    rows.append(method+' & '+' & '.join(vals)+' \\\\')
pred_header=r'''\begin{tabular}{lrrrr}\toprule
方法 & 训练交叉熵 & 测试交叉熵 & 测试准确率 & 训练时间/s\\\midrule
'''
(root/'results/predictor_table.tex').write_text(pred_header+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}'+'\n',encoding='utf-8')

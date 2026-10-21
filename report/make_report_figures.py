"""Create compact report figures from existing aggregate Project 1 outputs only."""
from pathlib import Path
import csv
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FIG = ROOT / 'report' / 'figures'
FIG.mkdir(exist_ok=True)

def read(name):
    with (OUT / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.dpi': 250})
def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

g = nx.read_graphml(OUT / 'graph_main.graphml')
part = {r['category']: int(r['leiden']) for r in read('communities.csv')}
keep = set()
for v in g:
    for u in sorted(g[v], key=lambda u: (-g[v][u]['weight'], u))[:5]:
        keep.add(tuple(sorted((u, v))))
h = nx.Graph()
h.add_nodes_from(g.nodes(data=True))
h.add_edges_from((u, v, g[u][v]) for u, v in sorted(keep))
h.remove_nodes_from(list(nx.isolates(h)))
pos = nx.spring_layout(h, seed=0, weight=None, iterations=300, k=1.6/np.sqrt(len(h)))
groups = [(2, 'Cardiometabolic (62)', '#2673bd'),
          (3, 'Hematology-associated (51)', '#159879'),
          (4, 'Cancer-associated (44)', '#d66337'),
          (9, 'Mental health (21)', '#d69a00'),
          (1, 'Musculoskeletal (72)', '#328437'),
          (0, 'Skin / eye dominant (73)', '#c56a9b'),
          (5, 'Digestive (33)', '#7564ba')]
colors = {c: col for c, _, col in groups}
fig = plt.figure(figsize=(7.1, 3.2))
ax = fig.add_axes([0.0, 0.01, 0.68, 0.98]); ax.set_axis_off()
nx.draw_networkx_edges(h, pos, ax=ax, width=0.22, alpha=0.22, edge_color='#8b969c')
nx.draw_networkx_nodes(h, pos, ax=ax,
    node_size=[5+95*float(g.nodes[v]['prevalence']) for v in h],
    node_color=[colors.get(part[v], '#bfc4c7') for v in h],
    linewidths=0.25, edgecolors='white')
for v in ['I50', 'N18', 'D70', 'C79', 'F32', 'F41', 'M25', 'L82', 'K31']:
    if v in pos:
        ax.annotate(v, pos[v], xytext=(2, 4), textcoords='offset points', fontsize=6,
                    bbox={'facecolor':'white', 'edgecolor':'none', 'pad':0.2, 'alpha':0.85})
legend = [Line2D([], [], marker='o', ls='', color=col, markersize=5, label=name)
          for _, name, col in groups]
legend.append(Line2D([], [], marker='o', ls='', color='#bfc4c7', markersize=5,
                     label='Other communities (121)'))
fig.legend(handles=legend, loc='center left', bbox_to_anchor=(0.68, 0.55),
           frameon=False, fontsize=7, labelspacing=1.05)
fig.text(0.71, 0.14, '477 nodes / 11,051 edges\nFull graph used for metrics', fontsize=7,
         color='#4f5a61', linespacing=1.5)
save(fig, 'community_network')

st = {(r['pair'],r['category']):float(r['share_in_top10']) for r in read('stability_bridges.csv')}
chosen = [('cardiometabolic-cancer_hematologic','N17','CM-H  N17  Kidney failure'),
          ('cardiometabolic-cancer_hematologic','E87','CM-H  E87  Fluid/electrolytes'),
          ('cardiometabolic-cancer_hematologic','D63','CM-H  D63  Anemia'),
          ('cardiometabolic-cancer_hematologic','J96','CM-H  J96  Respiratory failure'),
          ('cardiometabolic-mental_health','F06','CM-MH  F06  Mental disorders')]
vals = [st.get((pair,code),0) for pair,code,_ in chosen]
fig, ax = plt.subplots(figsize=(3.35, 2.0))
ax.barh(range(len(chosen)), vals, color=['#2673bd']*4+['#d69a00'], height=0.58)
ax.set_yticks(range(len(chosen)), [label for _,_,label in chosen], fontsize=6.5)
ax.invert_yaxis(); ax.set_xlim(0,1.13)
for i,v in enumerate(vals):ax.text(v+0.025,i,f'{round(v*30)}/30',va='center',fontsize=7)
ax.set_xticks([0,.5,1], ['0','0.5','1.0'])
ax.set_xlabel('Fraction in pair-specific top ten',fontsize=7)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
fig.tight_layout(pad=0.4)
save(fig,'bridge_stability')

fig, axes = plt.subplots(1,2,figsize=(7.1,2.65))
ranges = read('subgroup_edge_permutation_sex_significant.csv')
sex_flag = {tuple(sorted((r['u'],r['v']))):r['sex_specific']=='true' for r in ranges}
age_flags = {tuple(sorted((r['u'],r['v']))):r['sex_specific']=='true'
             for r in read('subgroup_edge_permutation_age_significant.csv')}
for ax, tag, flags, title, xlab, ylab in [
    (axes[0],'sex',sex_flag,'(a) Recorded sex','Phi in male patients','Phi in female patients'),
    (axes[1],'age',age_flags,'(b) Age at first eligible diagnosis','Phi in patients aged 65+','Phi in patients under 65')]:
    rows=read(f'subgroup_edge_permutation_{tag}.csv')
    x=np.array([float(r['phi_b']) for r in rows]); y=np.array([float(r['phi_a']) for r in rows])
    sig=np.array([float(r['p_fdr'])<.05 for r in rows])
    flag=np.array([flags.get(tuple(sorted((r['u'],r['v']))),False) for r in rows])
    ax.plot([-.12,1],[-.12,1],color='#81909b',lw=.65,zorder=0)
    for mask,col,size,label in [(~sig,'#b7bfc5',2,'Not selected'),
                              (sig&flag,'#d69a00',11,'Reproductive/sex-organ code'),
                              (sig&~flag,'#216bb0',11,'Other selected edge')]:
        ax.scatter(x[mask],y[mask],s=size,c=col,alpha=.65,linewidths=0,label=label,rasterized=True)
    ax.set_xlim(-.12,1);ax.set_ylim(-.12,1)
    ax.set_title(title,loc='left',fontsize=8)
    ax.set_xlabel(xlab,fontsize=7);ax.set_ylabel(ylab,fontsize=7)
    ax.tick_params(labelsize=7)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=6.5,
           bbox_to_anchor=(.5,-.015),markerscale=1.1)
fig.tight_layout(rect=[0,.085,1,1],pad=.5,w_pad=1.8)
save(fig,'demographic_edges')
print('Created three report figures (PDF and PNG).')

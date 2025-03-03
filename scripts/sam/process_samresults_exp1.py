# quick script to process stats.json results from SAM viewer

# experiment 1.
# 10 random cases, 2 slices per case
# compare speed and accuracy of single-point, manual bbox and BLAST-bbox prompts for SAM

import os
import json
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pandas as pd
import json
import copy
import seaborn as sns

datadir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases/SAM_BraTS_2024/brats2nifti'
casedirs = os.listdir(datadir)
casedirs = sorted([c for c in casedirs if c.startswith('M')])
casedirs2 = copy.copy(casedirs)
d = {}
for c in casedirs:
    studydirs = os.listdir(os.path.join(datadir,c))
    studydirs = [s for s in studydirs if s.startswith('S')]
    for s in studydirs:
        sdir = os.path.join(datadir,c,s)
        print(sdir)
        statsfile = os.path.join(sdir,'stats.json')
        if os.path.exists(statsfile):
            with open(statsfile,'r') as fp:
                sdict = json.load(fp)
            d[c] = sdict
        else:
            print('No stats files, skipping...')
            casedirs2.remove(c)

casedirs = casedirs2
prompts = list(d[c].keys())
res = {'dsc':{p:[] for p in prompts},'speed':{p:[] for p in prompts}}
for c in casedirs:
    print(c)
    for p in prompts:
        for r in ['roi1','roi2']:
            res['dsc'][p].append(d[c][p][r]['stats']['dsc']['TC'])
            res['speed'][p].append(d[c][p][r]['stats']['elapsedtime'])

mean_res = {'dsc':{},'speed':{}}
se_res = {'dsc':{},'speed':{}}
for stat in ['dsc','speed']:
    for p in prompts:
        mean_res[stat][p] = np.mean(res[stat][p])
        se_res[stat][p] = np.std(res[stat][p]) / np.sqrt(len(res[stat][p]))

pass
defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig,ax = plt.subplots(2,2,figsize=(6,6))
# plt.clf()
m = [mean_res['dsc'][k] for k in prompts]
se = [se_res['dsc'][k] for k in prompts]
plt.sca(ax[0,0])
ax[0,0].cla()
for i,p in enumerate(prompts):
    plt.errorbar(p,m[i],yerr=se[i],fmt='+',c=defcolors[i])
ax[0,0].set_ylim((0,1))
plt.ylabel('mean DSC+/- se')
plt.xticks(fontsize=8)

plt.sca(ax[0,1])
ax[0,1].cla()
sdata = {k:res['dsc'][k] for k in prompts}
sns.boxplot(data=sdata)
plt.ylabel('DSC')
plt.xticks(fontsize=8)

m = [mean_res['speed'][k] for k in prompts]
se = [se_res['speed'][k] for k in prompts]
plt.sca(ax[1,0])
ax[1,0].cla()
for i,p in enumerate(prompts):
    plt.errorbar(p,m[i],yerr=se[i],fmt='+',c=defcolors[i])
ax[1,0].set_ylim((0,25))
plt.ylabel('mean time +/- se')
plt.xticks(fontsize=8)

plt.sca(ax[1,1])
ax[1,1].cla()
sdata = {k:res['speed'][k] for k in prompts}
sns.boxplot(data=sdata)
ax[1,1].set_ylim((0,25))
plt.ylabel('time (sec)')
plt.xticks(fontsize=8)

plt.tight_layout()
plt.show(block=False)
plt.savefig(os.path.join(datadir,'results.png'))
pass
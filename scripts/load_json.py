# demo script to load results from 4panel viewer

import os
import json
import matplotlib.pyplot as plt
import base64
import imageio
import numpy as np
import pandas as pd

if os.name == 'posix':
    outputpath = '/media/jbishop/WD4/brainmets/sunnybrook/radnec/dicom2nifti_4panel/M0001'
else:
    outputpath = 'C:\\Users\\Chris Heyn Lab\\data\\dicom2nifti_4panel\\M0001'
filename = os.path.join(outputpath,'M0001.json')
tmpfile = os.path.join(outputpath,'tmpfile.png')

with open(filename) as fp:
    d = json.load(fp)

# print(d)

nstudy = len(d['images'])
nmeasure = len(d['images'][str(0)])
ik = ['t1+','flair','dwi','adc']
ikeys = []
ikeys.append([k+'0' for k in ik])
ikeys.append([k+'1' for k in ik])
nimage = len(ikeys[0])
fig,axs = plt.subplot_mosaic([ikeys[0],ikeys[1],['dwell']*4,['measurement']*4,['transcript']*4],
                                height_ratios=[1,1,1,1,1],
                                figsize=(6,8),dpi=96)
plt.show(block=False)

for s in range(nstudy):
    for i in range(nimage):
        iidx = s*nimage + i + 1
        imgb = base64.b64decode(d['images'][str(s)][0][ikeys[s][i][:-1]].encode('utf8')) # max dwelltime slice only
        with open(tmpfile,'wb') as fp:
            fp.write(imgb)
            img = imageio.v3.imread(tmpfile)
            fp.close()
            os.remove(tmpfile)
        axs[ikeys[s][i]].imshow(img,cmap='gray',origin='lower',aspect=1)
        axs[ikeys[s][i]].xaxis.set_tick_params(labelbottom=False)
        axs[ikeys[s][i]].yaxis.set_tick_params(labelleft=False)
        axs[ikeys[s][i]].set_xticks([])
        axs[ikeys[s][i]].set_yticks([])        
        # axs[ikeys[s][i]].axis('off')
        if i == 0:
            axs[ikeys[s][i]].set_title('study:{:d}, slice:{:d}'.format(s,int(d['dwell'][0][0])))

axs['dwell'].bar(list(map(str,map(int,d['dwell'][0]))),list(d['dwell'][1]))
axs['dwell'].set_ylabel('time (s)')

nmeas = 0
plt.sca(axs['measurement'])
axs['measurement'].axis('off')
for s in range(2):
    if d['measurement'][str(s)] is not None:
        nmeas += len(d['measurement'][str(s)])
ytext = 0.8
for s in range(2):
    if d['measurement'][str(s)] is not None:
        for m in range(len(d['measurement'][str(s)])):
            t = 'study={}, slice={}, length={:.1f}'.format(s,d['measurement'][str(s)][m]['slice'],d['measurement'][str(s)][m]['l'])
            plt.text(0,ytext,t)
            ytext -= .2

ytext = -.5
plt.sca(axs['transcript'])
axs['transcript'].axis('off')
if d['transcript'] is not None:
    transcript = ' '.join(d['transcript'])+'.'
    plt.text(0,ytext,transcript,wrap=True)
plt.savefig(os.path.join(outputpath,'summary.png'))
a=1
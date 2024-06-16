# demo script to load results from 4panel viewer

import os
import json
import matplotlib.pyplot as plt
import base64
import struct
import imageio

outputpath = '/media/jbishop/WD4/brainmets/sunnybrook/radnec/dicom2nifti_4panel/M0001'
filename = os.path.join(outputpath,'M0001.json')
tmpfile = os.path.join(outputpath,'tmpfile.png')

with open(filename) as fp:
    d = json.load(fp)

# print(d)

nstudy = len(d['images'])
ik = ['t1+','flair','dwi','adc']
ikeys = []
ikeys.append([k+'0' for k in ik])
ikeys.append([k+'1' for k in ik])
nimage = len(ikeys[0])
# fig = plt.figure(1,figsize=(8,11))
fig,axs = plt.subplot_mosaic([ikeys[0],ikeys[1],['.']*4],
                                height_ratios=[1,1,8],
                                figsize=(8,11),dpi=96)

for s in range(nstudy):
    for i in range(nimage):
        iidx = s*nimage + i + 1
        # ax = plt.subplot(nstudy,nimage,iidx)
        imgb = base64.b64decode(d['images'][str(s)][ikeys[s][i][:-1]].encode('utf8'))
        with open(tmpfile,'wb') as fp:
            fp.write(imgb)
            img = imageio.v3.imread(tmpfile)
            fp.close()
            os.remove(tmpfile)
        axs[ikeys[s][i]].imshow(img,cmap='gray',origin='lower',aspect=1)
        # _ = axs[ikeys[s][i]].imshow(img)
a=1
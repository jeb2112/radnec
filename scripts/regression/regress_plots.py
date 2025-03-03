# plots for temporal regression of 1d spatial data

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_data():

    # default colors for plots
    defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # some hard-coded limits for plots
    plotdata  = {
        "CT":
            {
                "d":(0,200),
                "d_norm":(-2,2),
                "d_diff":(-50,50),
                "d_norm_diff":(-1,1)
            },
        "MR":
            {
                "d":(0,500),
                "d_norm":(-2,2),
                "d_diff":(-100,100),
                "d_norm_diff":(-1,1)
            }
        }

    return defcolors,plotdata

# regression scatter plots
def plot_regression(xdata,ydata,resid,xfit,yfit,ci,rd,datadir,block=False,save=False,tag=''):
    _,plotdata = get_data()
    fig = plt.figure(1,figsize=(5,3))
    fig.clf()
    ax2 = plt.subplot(1,2,1)
    ax2.cla()
    ax2.plot(xfit,yfit*0,'b')
    # this tuple is a string in json
    plt.xlim(plotdata[rd['dcmtype']]['d_norm'])
    plt.xlabel('time 0')
    plt.ylabel('residual')
    ax2.fill_between(np.ravel(xfit), ci, -ci, color="#0277bd", edgecolor=None,alpha=0.5,zorder=2)
    ax2.set_aspect('equal')
    plt.scatter(xdata[::1000],resid[::1000],s=1,c='r',zorder=1)

    ax3 = plt.subplot(1,2,2)
    ax3.cla()
    ax3.plot(xfit,yfit,'b')
    plt.xlabel('time 0')
    plt.ylabel('time 1')
    ax3.fill_between(np.ravel(xfit), yfit + ci, yfit - ci, color="#0277bd", edgecolor=None,alpha=0.5,zorder=2)
    ax3.set_aspect('equal')
    plt.scatter(xdata[::1000],ydata[::1000],s=1,c='r',zorder=1)
    plt.xlim(plotdata[rd['dcmtype']]['d_norm'])
    plt.ylim(plotdata[rd['dcmtype']]['d_norm'])
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(datadir,'regression{:s}.png'.format(tag)))
    else:
        plt.show(block=block)


def plot_norm(nvals,datadir,block=False,save=False):
    defcolors,_ = get_data()
    plt.clf()
    xlim = (np.min([nvals['t0']['slim'][0],nvals['t1r']['slim'][0]])-100,np.max([nvals['t0']['slim'][1],nvals['t1r']['slim'][1]])+100)
    for i,dt in enumerate(['t0','t1r']):
        plt.subplot(1,2,i+1)
        plt.plot(nvals[dt]['counts'][0],nvals[dt]['q'],'+',c=defcolors[i])
        plt.plot(nvals[dt]['counts'][0],nvals[dt]['spl_q'],c=defcolors[i])
        plt.hlines([.2,.8],xlim[0],xlim[1],colors='k',linestyles='dotted')
        plt.xlim(xlim)
        plt.title(dt)
    if save:
        plt.savefig(os.path.join(datadir,'normalization.png'))
    else:
        plt.show(block=block)



def plot_mask(block=False,save=False):
    # check results
    plt.figure(10)
    plt.subplot(1,2,1),plt.cla()
    plt.imshow(roi_mask[plotslice])
    plt.subplot(1,2,2),plt.cla()
    plt.imshow(dset['t0'][d][plotslice],vmin=0,vmax=50,cmap='gray')
    if not save:
        plt.show(block=block)

# images and histograms
def plot_regimage(reg_image,diff_image,dset,rd,datadir,block=False,save=False,ct_offs=0):

    nplot = 1 + len(list(reg_image.keys()))
    # difference images
    defcolors,plotdata = get_data()
    vlims = plotdata[rd['dcmtype']]['d_norm_diff']
    plt.figure(10,figsize=(8,4))
    for i,m in enumerate(reg_image.keys()):
        plt.subplot(1,nplot,i+1)
        plt.imshow(reg_image[m][rd['plotslice']],vmin=vlims[0],vmax=vlims[1],cmap='gray')
        plt.title('regression '+m)
        plt.xticks((0,np.shape(reg_image[m])[1]-1))
        plt.yticks(())
    plt.subplot(1,nplot,nplot)
    plt.imshow(diff_image[rd['plotslice']]-ct_offs,vmin=vlims[0],vmax=vlims[1],cmap='gray')
    plt.title('subtraction')
    plt.xticks((0,np.shape(reg_image['_ransac'])[1]-1))
    plt.yticks(())
    plt.savefig(os.path.join(datadir,'diff_images.png'))

    # 1,2,1 difference histograms
    plt.figure(11),plt.clf()
    plt.subplot(1,2,1),plt.cla()
    rplot = reg_image['_ransac'][rd['plotslice']]
    rplot_ros = np.where(rplot != -ct_offs)
    plt.hist(np.ravel(rplot[rplot_ros]),50,range=(vlims[0],vlims[1]),density=True,color=defcolors[0],alpha=0.5)
    dplot = diff_image[rd['plotslice']]
    dplot_ros = np.where(dplot != -ct_offs)
    plt.hist(np.ravel(dplot[dplot_ros])-ct_offs,50,range=(vlims[0],vlims[1]),density=True,color=defcolors[1],alpha=0.5)
    plt.legend(('ransac','difference'),fontsize=7,loc='upper left')
    plt.title('time 1-0 difference',{'size':7})

    # 1,2,2 source histograms
    plt.subplot(1,2,2),plt.cla()
    xylims = plotdata[rd['dcmtype']]['d_norm']
    d0plot = dset['t0']['d_norm'][rd['plotslice']]
    d0_ros = np.where(d0plot != -ct_offs)
    d1plot = dset['t1r']['d_norm'][rd['plotslice']]
    d1_ros = np.where(d1plot != -ct_offs)
    plt.hist(np.ravel(d0plot[d0_ros]),50,range=(xylims[0],xylims[1]),density=True,color=defcolors[0],alpha=0.5)
    plt.hist(np.ravel(d1plot[d1_ros])-ct_offs,50,range=(xylims[0],xylims[1]),density=True,color=defcolors[1],alpha=0.5)
    plt.legend(('time 0','time 1'),fontsize=7,loc='upper left')
    plt.title('Input time 0,1 images',{'size':7})

    plt.suptitle('slice {}'.format(rd['plotslice']))


    if save:
        plt.savefig(os.path.join(datadir,'histograms.png'))
    else:
        plt.show(block=block)





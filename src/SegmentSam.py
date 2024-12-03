
import numpy as np
import os
import copy
import time

import matplotlib.pyplot as plt
from cProfile import Profile
from pstats import SortKey,Stats


#######################################
# methods for tumour segmenation by SAM
#######################################

# by default, SAM output is TC even as BLAST prompt input derived from t1+ is ET. because BLAST TC is 
# a bit arbitrary, not using it as the SAM prompt. So, layer arg here defaults to 'TC'
def segment_sam(self,roi=None,dpath=None,model='SAM',layer=None,tag='',prompt='bbox',orient=None,remote=False,session=None):
    print('SAM segment tumour')
    if roi is None:
        roi = self.ui.currentroi
    if layer is None:
        layer = self.layerROI.get()

    if dpath is None:
        dpath = os.path.join(self.ui.data[self.ui.s].studydir,'sam')
        if not os.path.exists(dpath):
            os.mkdir(dpath)

    if orient is None:
        orient = ['ax','sag','cor']

    if os.name == 'posix':
        if session is not None: # aws cloud

            casedir = self.ui.caseframe.casedir.replace(self.config.UIlocaldir,self.config.UIawsdir)
            studydir = self.ui.data[self.ui.s].studydir.replace(self.config.UIlocaldir,self.config.UIawsdir)
            dpath_remote = os.path.join(studydir,'sam')

            # run SAM
            for p in orient:
                command = 'python /home/ec2-user/scripts/main_sam_hf.py  '
                command += ' --checkpoint /home/ec2-user/sam_models/' + self.ui.config.SAMModelAWS
                command += ' --input ' + casedir
                command += ' --prompt ' + prompt
                command += ' --layer ' + layer
                command += ' --tag ' + tag
                command += ' --orient ' + p
                res = session.run_command2(command,block=False)

        else: # local
            for p in orient:
                if False: # standalone script
                    command = 'python scripts/main_sam_hf.py  '
                    command += ' --checkpoint /media/jbishop/WD4/brainmets/sam_models/' + self.ui.config.SAMModel
                    command += ' --input ' + self.ui.caseframe.casedir
                    command += ' --prompt ' + prompt
                    command += ' --layer ' + layer
                    command += ' --tag ' + tag
                    command += ' --orient ' + p
                    res = os.system(command)
                else: # run in viewer 
                    with Profile() as profile:
                        mfile = '/media/jbishop/WD4/brainmets/sam_models/' + self.ui.config.SAMModel
                        self.ui.sam.main(checkpoint=mfile,
                                        input = self.ui.caseframe.casedir,
                                        prompt = prompt,
                                        layer = layer,
                                        tag = tag,
                                        orient = p)
                        print('Profile: segment_sam, local')
                        (
                            Stats(profile)
                            .strip_dirs()
                            .sort_stats(SortKey.TIME)
                            .print_stats(25)
                        )

    elif os.name == 'nt': # not implemented yet

        if session is not None: # aws cloud

            casedir = self.ui.caseframe.casedir.replace(self.config.UIlocaldir,self.config.UIawsdir)
            studydir = self.ui.data[self.ui.s].studydir.replace(self.config.UIlocaldir,self.config.UIawsdir)
            dpath_remote = os.path.join(studydir,'sam')

            # run SAM
            for p in orient:
                command = 'python /home/ec2-user/scripts/main_sam_hf.py  '
                command += ' --checkpoint /home/ec2-user/sam_models/' + self.ui.config.SAMModelAWS
                command += ' --input ' + casedir
                command += ' --prompt ' + prompt
                command += ' --layer ' + layer
                command += ' --tag ' + tag
                command += ' --orient ' + p
                res = session.run_command2(command,block=False)

        else: # local
            for p in orient:
                with Profile() as profile:
                    self.ui.sam.main(checkpoint=os.path.join(os.path.expanduser('~'),'data','sam_models',self.ui.config.SAMModel),
                                    input = self.ui.caseframe.casedir,
                                    prompt = prompt,
                                    layer = layer,
                                    tag = tag,
                                    orient = p)
                    print('Profile: segment_sam, local')
                    (
                        Stats(profile)
                        .strip_dirs()
                        .sort_stats(SortKey.TIME)
                        .print_stats(25)
                    )

    return

# upload prompts to remote
def put_prompts_remote(self,session=None,do2d=True):
    dpath = os.path.join(self.ui.data[self.ui.s].studydir,'sam')
    studydir = self.ui.data[self.ui.s].studydir.replace(self.config.UIlocaldir,self.config.UIawsdir)
    dpath_remote = os.path.join(studydir,'sam')

    for d in ['ax','sag','cor','predictions_nifti']:
        if False:
            try:
                session.sftp.remove_dir(os.path.join(dpath_remote,d))
            except FileNotFoundError:
                pass
            except OSError as e:
                raise e
            except Exception as e:
                pass
            session.sftp.mkdir(os.path.join(dpath_remote,d))

    for d in ['ax','sag','cor']:
        if do2d:
            if True: # use paramiko. 
                localpath = os.path.join(dpath,d)
                remotepath = os.path.join(dpath_remote,d) # note here d must exist but be empty
                session.sftp.put_dir(localpath,remotepath)
                pass
            else: # use system
                localpath = os.path.join(dpath,d)
                remotepath = os.path.join(dpath_remote) # note here d does not exist and is copied
                command = 'scp -i ~/keystores/aws/awstest.pem -r ' + localpath + ' ec2-user@ec2-35-183-0-25.ca-central-1.compute.amazonaws.com:/' + remotepath
                os.system(command)
        else:
            for d2 in ['images','prompts']:
                localpath = os.path.join(dpath,d,d2)
                localfile = os.path.join(localpath,'png.tar')
                remotepath = os.path.join(dpath_remote,d,d2)
                remotefile = os.path.join(remotepath,'png.tar')
                command = 'mkdir -p ' + remotepath
                session.run_command2(command,block=True)
                session.sftp.put(localfile,remotefile)
                command = 'tar -xvf ' + remotefile + ' -C ' + remotepath
                command += '; rm ' + remotefile
                session.run_command2(command,block=False)
            for d2 in ['predictions']:
                remotepath = os.path.join(dpath_remote,d,d2)
                command = 'mkdir -p ' + remotepath
                session.run_command2(command,block=False)                    

# get the SAM results from remote.
def get_predictions_remote(self,tag='',session=None):

    dpath = os.path.join(self.ui.data[self.ui.s].studydir,'sam')
    localpath = os.path.join(dpath,'predictions_nifti')
    studydir = self.ui.data[self.ui.s].studydir.replace(self.config.UIlocaldir,self.config.UIawsdir)
    dpath_remote = os.path.join(studydir,'sam')
    remotepath = os.path.join(dpath_remote,'predictions_nifti')

    # poll until results are complete
    while True:
        command = 'ls ' + remotepath
        res = session.run_command(command)
        if len(res[0].split()) == 3:
            break
        else:
            time.sleep(.1)

    # download results
    start_time = time.time()
    session.sftp.get_dir(remotepath,localpath)

    # clean up results dir
    command = 'rm -rf ' + remotepath
    session.run_command2(command,block=False)

    return time.time() - start_time

# load the results of SAM
def load_sam(self,layer=None,tag='',prompt='bbox',do_ortho=False,do3d=True):
            
    if layer is None:
        layer = self.layerROI.get()

    # shouldn't be needed?
    self.update_roinumber_options()

    roi = self.ui.currentroi
    rref = self.ui.rois['sam'][self.ui.s][roi]
    if do_ortho:
        img_ortho = {}
        for p in ['ax','sag','cor']:
            fname = layer+'_sam_' + prompt + '_' + tag + '_' + p + '.nii.gz'
            img_ortho[p],_ = self.ui.data[self.ui.s].loadnifti(fname,
                                                        os.path.join(self.ui.data[self.ui.s].studydir,'sam','predictions_nifti'),
                                                        type='uint8')
        # in 3d take the AND composite segmentation
        if do3d:
            img_comp = img_ortho['ax']
            for p in ['sag','cor']:
                img_comp = (img_comp) & (img_ortho[p])
            rref.data[layer] = copy.deepcopy(img_comp)
        # in 2d use the individual slice segmentations by OR
        else:
            img_comp = img_ortho['ax']
            for p in ['sag','cor']:
                img_comp = (img_comp) | (img_ortho[p])
            rref.data[layer] = copy.deepcopy(img_comp)


    else:
        fname = layer+'_sam_' + prompt + '_' + tag + '_ax.nii.gz'
        rref.data[layer],_ = self.ui.data[self.ui.s].loadnifti(fname,
                                                        os.path.join(self.ui.data[self.ui.s].studydir,'sam','predictions_nifti'),
                                                        type='uint8')
    # create a combined seg mask from the three layers
    # using nnunet convention for labels
    rref.data['seg'] = 2*rref.data['TC'] + 1*rref.data['WT']
    # need to add this to updateData() or create similar method
    if False:
        self.ui.data[self.ui.s].mask['sam'][layer]['d'] = copy.deepcopy(self.ui.roi[self.ui.s][roi].data[layer])
        self.ui.data[self.ui.s].mask['sam'][layer]['ex'] = True

    # shouldn't be needed?
    self.update_roinumber_options()

    return 



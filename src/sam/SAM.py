import numpy as np
import os
import re
import matplotlib.pyplot as plt
import shutil
import copy
from cProfile import Profile
from pstats import SortKey,Stats
import time

from sklearn.metrics import f1_score, precision_score
from scipy.spatial.distance import dice,directed_hausdorff
from scipy.stats import logistic
import skimage
from skimage.io import imread,imsave
from skimage.transform import resize
import nibabel as nb
import PIL 
import cv2

from transformers import SamModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# to suppress symlink warnings in windows 11
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from src.sam.SAMProcessing import SAMProcessing
from src.sam.SAMMisc import *


class SAM():
    def __init__(self,ui=None,remote=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.remote = remote
        self.ui = ui
        self.samp = None

    ################
    # prep functions
    ################

    # by default, SAM output is TC even as BLAST prompt input derived from t1+ is ET. because BLAST TC is 
    # a bit arbitrary, not using it as the SAM prompt. So, layer arg here defaults to 'TC'
    def segment_sam(self,roi=None,dpath=None,model='SAM',layer=None,tag='',prompt='bbox',orient=None,remote=False,session=None):
        print('SAM segment tumour')
        if roi is None:
            roi = self.ui.currentroi
        if layer is None:
            layer = self.ui.roiframe.roioverlayframe.layerROI.get()

        if dpath is None:
            dpath = os.path.join(self.ui.data[self.ui.s].studydir,'sam')
            if not os.path.exists(dpath):
                os.mkdir(dpath)

        if orient is None:
            orient = ['ax','sag','cor']

        if os.name == 'posix':
            if session is not None: # aws cloud. not updated yet

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
                        mfile = '/media/jbishop/WD4/brainmets/sam_models/' + self.ui.config.SAMModel
                        self.run_sam(checkpoint=mfile,
                                        input = self.ui.caseframe.casedir,
                                        prompt = prompt,
                                        layer = layer,
                                        tag = tag,
                                        orient = p,
                                        debug=True)
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
                        self.run_sam(checkpoint=os.path.join(os.path.expanduser('~'),'data','sam_models',self.ui.config.SAMModel),
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
            layer = self.ui.roiframe.roioverlayframe.layerSAM.get()

        # shouldn't be needed?
        self.ui.roiframe.update_roinumber_options()

        roi = self.ui.currentroi
        rref = self.ui.rois['sam'][self.ui.s][roi]
        if do_ortho:
            img_ortho = {'sam':{},'sam_raw':{}}
            for p in ['ax','sag','cor']:
                for m in img_ortho.keys():
                    fname = layer+'_' + m +'_' + prompt + '_' + tag + '_' + p + '.nii.gz'
                    if m == 'sam_raw':
                        dtype = 'float'
                    else:
                        dtype = 'uint8'
                    img_ortho[m][p],affine = self.ui.data[self.ui.s].loadnifti(fname,
                                                                os.path.join(self.ui.data[self.ui.s].studydir,'sam','predictions_nifti'),
                                                                type=dtype)
                    # if padded for hugging face, re-crop here
                    img_ortho[m][p] = img_ortho[m][p][:self.ui.sliceviewerframe.dim[0],:self.ui.sliceviewerframe.dim[1],:self.ui.sliceviewerframe.dim[2]]

            # in 3d take the AND composite segmentation or average the probabilities
            if do3d:
                img_comp = img_ortho['sam']['ax']
                for p in ['sag','cor']: # binary recombination by AND
                    img_comp = (img_comp) & (img_ortho['sam'][p])
                if False: # CDF recombination incorporating some measure of correlation?
                    raw_comp = np.zeros_like(img_ortho['sam_raw']['ax'],dtype='float')
                else: # assuming independence
                    raw_comp = img_ortho['sam_raw']['ax']
                    for p in ['sag','cor']:
                        raw_comp *= img_ortho['sam_raw'][p]
                raw_comp = logistic.cdf(raw_comp)
                raw_comp = (raw_comp > 0.5).astype(np.uint8) * 255

                fname = layer+'_samcombined_' + prompt + '_' + tag + '.nii'
                fpath = os.path.join(self.ui.data[self.ui.s].studydir,'sam','predictions_nifti',fname)
                if self.ui.config.SAMRawCombine:
                    # re save the combined mask
                    self.ui.data[self.ui.s].writenifti(raw_comp,fpath, affine=affine, type='uint8')
                    rref.data[layer] = copy.deepcopy(raw_comp)
                else:
                    self.ui.data[self.ui.s].writenifti(img_comp,fpath, affine=affine, type='uint8')
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
            # if padded for hugging face, re-crop here
            rref.data[layer] = rref.data[layer][:,0:self.ui.sliceviewerframe.dim[1],0:self.ui.sliceviewerframe.dim[2]]

        # create a combined seg mask from the three layers
        # using nnunet convention for labels
        rref.data['seg'] = 2*rref.data['TC'] + 1*rref.data['WT']
        # need to add this to updateData() or create similar method
        if False:
            self.ui.data[self.ui.s].mask['sam'][layer]['d'] = copy.deepcopy(self.ui.roi[self.ui.s][roi].data[layer])
            self.ui.data[self.ui.s].mask['sam'][layer]['ex'] = True

        # shouldn't be needed?
        self.ui.roiframe.update_roinumber_options()

        return 


    ######################
    # prediction functions
    ######################

    def upsample_mask(self,pred_mask, target_size):
        return F.interpolate(pred_mask.unsqueeze(1), size=target_size, mode="nearest").squeeze(1)

    def forward_pass(self,model: SamModel, batch, prompt_type: PromptType, multi_mask=False):
        if prompt_type == PromptType.CONTROL_POINTS:
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_points=batch["input_points"],
                input_labels=batch["input_labels"],
                multimask_output=multi_mask,
            )
        elif prompt_type == PromptType.BOUNDING_BOX:
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_boxes=batch["input_boxes"],
                multimask_output=False,
            )
        return outputs

    def calc_datapoint_metrics(self,y_true_mask, y_pred_mask):
        y_true = y_true_mask.ravel()
        y_pred = y_pred_mask.ravel()
        hd =  max(directed_hausdorff(np.array(np.where(y_true_mask)).T,np.array(np.where(y_pred_mask)).T)[0],
                                                            directed_hausdorff(np.array(np.where(y_pred_mask)).T,np.array(np.where(y_true_mask)).T)[0])
        if hd == np.Inf:
            hd = np.NaN
        return {
            # "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "dice": 1-dice(y_true,y_pred),
            "hausdorff": hd

        }

    def predict_model(self,model: SamModel, batch, prompt_args, confidence_threshold=0.5):

        # Squeeze the 0th dim since prediction dataloader always has batch size 1
        # need to check this in a training context
        for k in batch.keys(): 
            batch[k] = batch[k].squeeze(0)
            
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            if True:
                outputs = self.forward_pass(model, batch, prompt_args['prompt_type'], multi_mask=prompt_args['multi_mask'])
            else:
                outputs = model(**batch, multimask_output=True)

        # squeeze extra dimensions. this appears to be the same for both bbox and points.
        raw_predicted_masks = outputs.pred_masks.squeeze((0, 1)).cpu()
        if prompt_args['prompt_type'] == 'bbox':
            ground_truth_mask = batch["ground_truth_mask"].squeeze(0).float().cpu()

        if False: # sigmoid processing
            predicted_masks = torch.sigmoid(self.upsample_mask(raw_predicted_masks, self.ui.sliceviewerframe.dim[1:])).cpu().numpy()
            predicted_masks = (sigmoid_mask > confidence_threshold).astype(np.uint8) * 255
        else: # huggingface default processing
            predicted_masks = self.samp.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), batch["original_sizes"].cpu(), batch["reshaped_input_sizes"].cpu())
            predicted_masks = predicted_masks[0].squeeze().cpu().numpy().astype(np.uint8) * 255
            raw_predicted_masks = self.upsample_mask(outputs.pred_masks.cpu().squeeze(),self.ui.sliceviewerframe.dim[1:]).cpu().numpy()
        # metrics = calc_datapoint_metrics(ground_truth_mask.numpy().astype(np.uint8), mask)
        metrics = None

        # pick best mask if multiple
        if len(np.shape(predicted_masks)) > 2:
            if False: # anecdotally, the highest iou is turning out to be the entire brain, not the lesion. 
                    # generally, the desired mask is looking like the 3rd mask. 
                    # but even the intermediate mask has sometimes higher iou. 
                    # however, turning multi-mask off makes a worse result.
                    # so not sure what the logic can be here
                confidence_idx = torch.argmax(outputs['iou_scores']).cpu()  # Index of the most confident mask
                mask = predicted_masks[confidence_idx]
            else:
                mask = predicted_masks[-1]
                raw_mask = raw_predicted_masks[-1]
        else:
            mask = predicted_masks
            raw_mask = raw_predicted_masks


        mask_comb = None
        if False:
            bbox = batch['input_boxes'][0][0].cpu().numpy().astype('int')
            bbox_shape = tuple(batch['reshaped_input_sizes'][0].cpu().numpy())
            mask_comb = predict_plot(ground_truth_mask.numpy().astype(np.uint8),mask,bbox,bbox_shape)

        return mask, raw_mask, metrics


    def predict_metrics(self,model: SamModel, dataloader: DataLoader, prompt_args, datadir=None):
        model.to(self.device)
        model.eval()

        metrics = []    
        
        for idx, batch in enumerate(dataloader):
            mask, raw_mask, batch_metrics = self.predict_model(model, batch, prompt_args)
            metrics.append(batch_metrics)

            if datadir is not None:
                ofile = os.path.join(datadir,'predictions','pred_mask_' + str(idx).zfill(5) + '.png')
                imsave(ofile,mask,check_contrast=False)
                ofile = os.path.join(datadir,'predictions','raw_pred_mask_' + str(idx).zfill(5) + '.tiff')
                cv2.imwrite(ofile,raw_mask)
                # imsave(ofile,raw_mask,check_contrast=False)

        return metrics

    def predict_plot(self,gt,pred,bbox,shape):
        comb_mask = np.zeros(shape+(3,),dtype='uint8')
        gt = resize(gt,shape,order=0).astype('uint8') * 1
        pred = resize(pred,shape,order=0).astype('uint8') * 1
        comb_mask[gt>0,0] = 255 # red
        comb_mask[pred>0,2] = 255 # blue
        rr,cc = skimage.draw.rectangle_perimeter(bbox[1::-1],end=bbox[:1:-1],shape=comb_mask.shape)
        comb_mask[rr,cc,1] = 255
        return comb_mask

    ################
    # misc functions
    ################
            
    def set_slice(self,idx,img_arr_3d,img_arr_2d,orient):
        if orient == 'ax':
            img_arr_3d[idx] = copy.deepcopy(img_arr_2d)
        elif orient == 'sag':
            img_arr_3d[:,:,idx] = copy.deepcopy(img_arr_2d)
        elif orient == 'cor':
            img_arr_3d[:,idx,:] = copy.deepcopy(img_arr_2d)

    def load_model_checkpoint(self,checkpoint_path, model: SamModel):
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Model checkpoint loaded from: {checkpoint_path}")

        return model

    # load a single nifti file
    def loadnifti(self,t1_file,dir,type=None):
        img_arr_t1 = None
        try:
            img_nb_t1 = nb.load(os.path.join(dir,t1_file))
        except IOError as e:
            print('Can\'t import {}'.format(t1_file))
            return None,None
        nb_header = img_nb_t1.header.copy()
        # nibabel convention will be transposed to sitk convention
        img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
        if type is not None:
            img_arr_t1 = img_arr_t1.astype(type)
        affine = img_nb_t1.affine
        return img_arr_t1,affine

    # write a single nifti file. use uint8 for masks 
    def writenifti(self,img_arr,filename,header=None,norm=False,type='float64',affine=None):
        img_arr_cp = copy.deepcopy(img_arr)
        if norm:
            img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
        # using nibabel nifti coordinates
        img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
        nb.save(img_nb,filename)
        if True:
            os.system('gzip --force "{}"'.format(filename))


    #############
    # script main
    #############

    def run_sam(self,checkpoint=None,input=None,tag=None,layer=None,orient=None,prompt='bbox',
             pretrained=None,output=None,debug=False):

        prompt_args = {'point':{    # Control points version:
            "prompt_type": PromptType.CONTROL_POINTS,
            "multi_mask": True,
            },
            'maskpoint':{ # variant of control points version derived from BLAST mask
            "prompt_type": PromptType.CONTROL_POINTS,
            "multi_mask": True,
            },
            'bbox':{ # Bounding boxes version:
            "prompt_type": PromptType.BOUNDING_BOX,
            "perturbation": 0,
            "padding": 3,
            "multi_mask": None
            }
        }

        if pretrained is not None:
            # getting a path error on this syntax
            model = SamModel.from_pretrained(pretrained)
        else:
            # this loads from cache after 1st download
            # this also gets loaded in huggingface, need to 
            # prevent this duplication
            if True:
                model = SamModel.from_pretrained(f"facebook/sam-vit-base")
        if checkpoint is not None:
            checkpoint_data = torch.load(checkpoint)
            if hasattr(checkpoint_data,'model_state_dict'):
                model_state_dict = checkpoint_data["model_state_dict"]
                model.load_state_dict(model_state_dict)
            else:
                model_state_dict = None
        else:
            model_state_dict = None
        if True:
            model.to(self.device) # this is performed in huggingface again duplication


        sfiles = os.listdir(input)
        if False:
            studydirs = [s for s in sfiles if os.path.isdir(os.path.join(input,s,'sam',orient,'images'))]
            if len(studydirs) == 0:
                raise FileNotFoundError('No images sub-directory found')
        else: # only coded for 1 actual dir right now
            if not os.path.isdir(os.path.join(self.ui.data[self.ui.s].studydir,'sam',orient,'images')):
                raise FileNotFoundError('No images sub-directory found')
            studydirs = [self.ui.data[self.ui.s].studydir]
        for s in studydirs: 
            # prepare output dir. When orthogonal planes are being predicted, 'ax' will be first.
            if output is None:
                outputdir = os.path.join(input,s,'sam','predictions_nifti')
            else:
                outputdir = output

            if not os.path.exists(outputdir):
                os.mkdir(outputdir)

            spath = os.path.join(input,s,'sam',orient)
            if not os.path.exists(spath):
                raise NotADirectoryError
            files = os.listdir(spath)
            if len(files) == 0:
                raise FileNotFoundError
            
            # load 1st image to get some of the image params
            img_files = os.listdir(os.path.join(spath,'images'))
            images = sorted([ f for f in img_files if re.match('img.*slice_[0-9]{3}',f) ])
            image_pil = PIL.Image.open(os.path.join(spath,'images',images[0]))
            # previously, the affines were encoded in the .png header when using the standalone
            # sam_hf.py script. now that SAM inference is implemented in the viewer, probably don't need this
            # arrangement anymore. 
            slicedim = int(image_pil.info['slicedim'])
            affine_enc = image_pil.info['affine'].encode()
            affine_dec = affine_enc.decode('unicode-escape').encode('ISO-8859-1')[2:-1]
            affine = np.reshape(np.frombuffer(affine_dec, dtype=np.float64),(4,4))

            eval_datadir = {'test':spath}
            self.samp = SAMProcessing(eval_datadir,model_dict=model_state_dict,
                                 model_size='base',
                                 prompt_type=prompt_args[prompt]['prompt_type'])
                                #  image_size=img_size)

            res = self.predict_metrics(model,self.samp.dataloaders['test'],prompt_args[prompt],datadir=spath)

            # gather the sam-predicted 2d slices into a nifti volume
            # with the torch DataLoader, the output masks for a set of image files are just in 
            # plain index order. map these back to the slice positions according to the filenames
            # of the input image files. it is thus assumed that the DataLoader is processing the image
            # files in a sorted order since it is batchsize=1 and no shuffling (to be verified)

            sam_predict = {'sam':None,'sam_raw':None}
            if orient == 'ax':
                sam_predict['sam'] = np.zeros((slicedim,)+image_pil.size[-1::-1],dtype='uint8')
                sam_predict['sam_raw'] = np.zeros((slicedim,)+image_pil.size[-1::-1],dtype='float')
            elif orient == 'sag':
                sam_predict['sam'] = np.zeros(image_pil.size[-1::-1]+(slicedim,),dtype='uint8')
                sam_predict['sam_raw'] = np.zeros(image_pil.size[-1::-1]+(slicedim,),dtype='float')
            elif orient == 'cor': 
                sam_predict['sam'] = np.zeros((image_pil.size[-1],slicedim,image_pil.size[0]),dtype='uint8')
                sam_predict['sam_raw'] = np.zeros((image_pil.size[-1],slicedim,image_pil.size[0]),dtype='float')
            image_pil.close()

            pred_files = os.listdir(os.path.join(spath,'predictions'))
            # binary, probability masks
            for ttag,mtag in zip(['sam','sam_raw'],['pred_mask','raw_pred_mask']):
                pred_masks = sorted([ f for f in pred_files if re.match(mtag,f) ])
                for i,m in zip(images,pred_masks):
                    slice = int(re.search('slice_([0-9]{3})',i).group(1))
                    mask_pil = PIL.Image.open(os.path.join(spath,'predictions',m))
                    mask = skimage.transform.resize(np.array(mask_pil),image_pil.size[-1::-1],order=0)
                    self.set_slice(slice,sam_predict[ttag],mask,orient)
                    mask_pil.close() 

                fname = '{}_{}_{}_{}_{}.nii'.format(layer,ttag,prompt,tag,orient)
                if ttag == 'sam':
                    if np.max(sam_predict[ttag]) > 1:
                        sam_predict[ttag][np.where(sam_predict[ttag])] = 1
                    self.writenifti(sam_predict[ttag],os.path.join(outputdir,fname),type=np.uint8,affine=affine)
                else:
                    self.writenifti(sam_predict[ttag],os.path.join(outputdir,fname),type=np.float32,affine=affine)

            # clean up working directories
            if not debug:
                shutil.rmtree(spath)
                for d in ['images','prompts','predictions']:
                    os.makedirs(os.path.join(spath,d),exist_ok=True)



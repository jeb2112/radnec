import numpy as np
import skimage
from skimage.io import imread,imsave
from skimage.transform import resize
import monai
import glob
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone, timedelta
import gc
from torch.utils.data import Dataset

class PromptType:  
    CONTROL_POINTS = "pts"  
    BOUNDING_BOX = "bbox"

class SAMDataset(Dataset):  
    def __init__(  
        self,   
        datadir,   
        processor,   
        prompt_type = PromptType.BOUNDING_BOX,  
        perturbation = 0,
        padding = 3,  
        image_size = (1024, 1024),  
        mask_size = (256, 256),  
    ):  
        # Asign all values to self  
        self.datadir = datadir
        self.dataset = os.listdir(os.path.join(self.datadir,'images'))
        self.processor = processor
        self.prompt_type = prompt_type
        self.image_size = image_size
        self.mask_size = mask_size

        # This is only to be used for BOUNDING_BOX prompts.
        self.perturbation = perturbation
        self.padding = padding

    def __len__(self):  
        return len(self.dataset)

    def __getitem__(self, idx):  


        # hard-coded file syntax
        try:
            ifile = glob.glob(os.path.join(self.datadir,'images','img_' + str(idx).zfill(5) + '_case_*_slice_???.png'))[0]
        except IndexError as e:
            print(e,idx)
            raise IndexError
        input_image = imread(ifile)

        # should be done by huggingface processor
        if False:
            input_image1 = skimage.transform.resize(input_image,self.image_size)
        else:
            input_image1 = input_image

        # huggingface transformers SamImageProcessor claims to support channel dim
        # first or last and infer from shape of data, but last didn't work and 
        # first did. 
        # standalone sam preprocess script outputs a gray-scale image which has to be tiled back to rgb
        # viewer code currently outputs a RGBA image, in the 3rd dim, which has to be switched to rgb, in the first dim
        if len(np.shape(input_image1)) == 3:
            input_image1 = np.tile(input_image1[:,:,0],(3,1,1))
        elif len(np.shape(input_image1)) == 2:
            input_image1 = np.tile(input_image1,(3,1,1))
            
        # subdir 'prompts' is hard-coded here
        # currently bounding box prompts are being stored as mask images, but this should be changed to be json or other text.
        if self.prompt_type == PromptType.BOUNDING_BOX:
            ifile = glob.glob(os.path.join(self.datadir,'prompts','img_' + str(idx).zfill(5) + '_case_*_slice_???.png'))[0]
            # again, masks prepared by viewer code are currently rgba, float. 
            ground_truth_mask = imread(ifile,as_gray=True).astype('uint8') * 255
            # note. skimage resize is one of the many resize algorithms that has an easy option for
            # nearest neighbour. so many others do not.
            ground_truth_mask = skimage.transform.resize(ground_truth_mask,self.mask_size,order=0)
            inputs = self._getitem_bbox(input_image1, ground_truth_mask)
            # this is a dummy mask which is just an image of the bbox
            inputs["ground_truth_mask"] = ground_truth_mask  
        # control point prompts are stored as json dict
        elif self.prompt_type == PromptType.CONTROL_POINTS:
            ifile = glob.glob(os.path.join(self.datadir,'prompts','pts_' + str(idx).zfill(5) + '_case_*_slice_???.json'))[0]
            with open(ifile,'r') as fp:
                ground_truth_pts = json.load(fp)
            inputs = self._getitem_ctrlpts(input_image1, ground_truth_pts=ground_truth_pts)  

        # debug plotting
        if False:

            # note matplotlib 3.9.0 vs code 1.92.2 plts stopped showing in debug console,
            # until using plt.show(block=True), which is the default corresponding to
            # plt.show(), but neither plt.show() nor plt.show(block=False work
            if self.prompt_type == PromptType.BOUNDING_BOX:
                check_image = np.copy(inputs['pixel_values'])[0]
                bbox = np.array(inputs['input_boxes'][0]).astype('int')
                check_mask = np.copy(ground_truth_mask).astype('uint8')

                if True: # additionally save to file. 
                    # 32 bit tiffs
                    if False:
                        vmax = np.max(check_image)
                        rr,cc = skimage.draw.rectangle_perimeter(bbox[1::-1],end=bbox[:1:-1],shape=check_image.shape)
                        check_image[rr,cc] = vmax
                        check_mask = resize(check_mask,np.shape(check_image),order=0).astype('float32') * vmax
                        check_comb = np.concatenate((check_image,check_mask),axis=1)
                        ofile = os.path.join(self.datadir,'datacheck','comb_' + str(idx).zfill(4) + '.tiff')
                        cv2.imwrite(ofile,check_comb)
                    else:
                    # 8 bit pngs
                        check_image2 = (skimage.transform.resize(input_image,np.shape(check_image))*255).astype('uint8')
                        check_mask = resize(check_mask,np.shape(check_image2),order=0).astype('uint8') * 255
                        rr,cc = skimage.draw.rectangle_perimeter(bbox[1::-1],end=bbox[:1:-1],shape=check_mask.shape)
                        check_image2[rr,cc] = 255
                        check_comb = np.concatenate((check_image2,check_mask),axis=1).astype('uint8')
                        ofile = os.path.join(self.datadir,'datacheck','comb_' + str(idx).zfill(4) + '.png')
                        imsave(ofile,check_comb)

            elif self.prompt_type == PromptType.CONTROL_POINTS:
                check_image = np.copy(inputs['pixel_values'])[0]
                pts = np.array(inputs['input_points'][0]).astype('int')

                # 8 bit pngs
                check_image2 = (skimage.transform.resize(input_image,np.shape(check_image))*255).astype('uint8')
                for p in pts:
                    check_image2[p[1]-4:p[1]+4,p[0]-4:p[0]+4] = np.array([255,0,0,255],dtype='uint8')
                ofile = os.path.join(self.datadir,'datacheck','comb_' + str(idx).zfill(4) + '.png')
                plt.figure(8)
                plt.imshow(check_image2)
                plt.show(block=False)
                # imsave(ofile,check_image2)

        return inputs
    
    # control points option. single point positive point, centroid of a mask, or multiple labelled points 
    def generate_input_points(self,mask=None,pts=None,dynamic_distance=True,erode=True):
        input_points = []
        if mask is not None:
            row,col = map(int,np.mean(np.where(mask==255),axis=1))
            # need [col,row]
            input_points.append([col,row]) 
            input_points = np.array(input_points)
            input_labels = np.array([1])
        elif pts is not None:
            # add dummy dimension for batch
            input_points = np.array([(x,y) for x,y in zip(pts['x'],pts['y'])])
            # dummy batch dimension? or samprocessor adds automatically
            # input_labels = np.expand_dims(np.array(pts['fg']),0)
            input_labels = np.array(pts['fg'])

        return  input_points,input_labels  

    def _getitem_ctrlpts(self, input_image, ground_truth_pts=None):  
        # Get control points prompt. See the GitHub for the source  
        # of this function, or replace with your own point selection algorithm.  
        input_points, input_labels = self.generate_input_points(  
            pts=ground_truth_pts
        )  
        # Samprocessor requires lists not np arrays
        input_points = [input_points.astype(float).tolist()]  
        input_labels = input_labels.tolist()  
        if False:
            input_labels = [[x] for x in input_labels] # teodoran code
        else:
            input_labels = [input_labels] # chatgpt code

        # Prepare the image and prompt for the model.  
        inputs = self.processor(  
            input_image,  
            input_points=input_points,  
            input_labels=input_labels,  
            return_tensors="pt"  
        )

        # pre-Remove batch dimension because it gets added back later.  
        # inputs = {k: v for k, v in inputs.items()}
        for k in inputs.keys(): 
            inputs[k] = inputs[k].squeeze(0)

        return inputs

    # bbox option
    def get_input_bbox(self,mask, perturbation=0):
        # Find minimum mask bounding all included mask points.
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add padding
        ydim,xdim = np.shape(mask)
        x_min = max(0,x_min - self.padding)
        y_min = max(0,y_min - self.padding)
        x_max = min(xdim-1,x_max + self.padding)
        y_max = min(ydim-1,y_max + self.padding)

        if perturbation:  # Add perturbation to bounding box coordinates.
            H, W = mask.shape
            x_min = max(0, x_min + np.random.randint(-perturbation, perturbation))
            x_max = min(W, x_max + np.random.randint(-perturbation, perturbation))
            y_min = max(0, y_min + np.random.randint(-perturbation, perturbation))
            y_max = min(H, y_max + np.random.randint(-perturbation, perturbation))

        bbox = [x_min, y_min, x_max, y_max]
        
        return bbox

    def _getitem_bbox(self, input_image, ground_truth_mask):  
        # Get bounding box prompt.  
        bbox = self.get_input_bbox(ground_truth_mask, perturbation=self.perturbation)

        # Prepare the image and prompt for the model.  
        inputs = self.processor(input_image, input_boxes=[[bbox]], return_tensors="pt")  
        inputs = {k: v.squeeze(0) for k, v in inputs.items()} # Remove batch dimension which the processor adds by default.

        return inputs
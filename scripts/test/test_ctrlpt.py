from transformers import SamProcessor, SamModel
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def upsample_mask(pred_mask, target_size):
    return F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0), size=target_size, mode="nearest").squeeze(1)

# Load the processor and model
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")

# Prepare inputs
input_points = [[[115, 92], [117, 108]]]
input_labels = [[1, 0]]

input_image = np.zeros((233,197,3),dtype=np.uint8)
input_image[20:210,20:177,:] = 128
input_image[88:96,111:118,:] = 255

# optionally crop to square aspect
if False:
    input_image = input_image[:197,:,:]

inputs = processor(
    images=input_image,  # Your image
    input_points=input_points,
    input_labels=input_labels,
    return_tensors="pt",
    mask_size=(197,233)
)

check_image = np.copy(inputs['pixel_values'])[0]
check_image = np.moveaxis(check_image,0,-1)
pts = np.array(inputs['input_points'][0]).astype('int')

# display the control points
for p in pts[0]:
    check_image[p[1]-4:p[1]+4,p[0]-4:p[0]+4,:] = np.array([255,0,0],dtype='uint8')

# Forward pass with multimask_output=False
outputs = model(**inputs, multimask_output=False)

# Retrieve masks
pred_mask = np.squeeze(outputs["pred_masks"])  # Check the shape and content of this tensor
pred_mask_sig = torch.sigmoid(upsample_mask(pred_mask, np.shape(input_image)[:2])).cpu().detach().numpy().squeeze()
thresh_mask = (pred_mask_sig > 0.5).astype(np.uint8) * 255

plt.figure(8)
plt.subplot(1,4,1)
plt.imshow(input_image)

plt.subplot(1,4,2)
plt.imshow(check_image)

plt.subplot(1,4,3)
plt.imshow(pred_mask.cpu().detach().numpy().squeeze())

plt.subplot(1,4,4)
plt.imshow(thresh_mask)
plt.show()
pass
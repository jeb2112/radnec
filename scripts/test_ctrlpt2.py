from PIL import Image
import requests

if False:
    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
    input_points = [[[2592, 1728],[2640,1820]]] # point location of the bee
    input_labels = [[1,1]]
    image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
else:
    img_url = "/home/jbishop/Pictures/img_00000_case_M00012_slice_085.png"
    input_points = [[[144,168]]]
    input_labels = [[1]]
    image = Image.open(img_url).convert("RGB")

from transformers import SamModel, SamProcessor
import torch

device = 'cuda'
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

inputs = processor(image, input_points=input_points, input_labels=input_labels,return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
confidence_idx = outputs['iou_scores'][0][0].cpu().numpy()  # Index of the most confident mask

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image')
mask_list = [masks[0][0][0].numpy(), masks[0][0][1].numpy(), masks[0][0][2].numpy()]

for i, mask in enumerate(mask_list, start=1):
    overlayed_image = np.array(image).copy()

    overlayed_image[:,:,0] = np.where(mask == 1, 255, overlayed_image[:,:,0])
    overlayed_image[:,:,1] = np.where(mask == 1, 0, overlayed_image[:,:,1])
    overlayed_image[:,:,2] = np.where(mask == 1, 0, overlayed_image[:,:,2])
    
    axes[i].imshow(overlayed_image)
    axes[i].set_title('Mask {}, iou={:.2f}'.format(i,confidence_idx[i-1]))
for ax in axes:
    ax.axis('off')

plt.show()
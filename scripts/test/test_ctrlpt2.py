# test script to evaluate SAM 

from PIL import Image
from transformers import SamModel, SamProcessor
import torch
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


img_url = "/home/jbishop/Pictures/img_00000_case_M00012_slice_085.png"
input_points = [[[144,168]]]
input_labels = [[1]]
image = Image.open(img_url).convert("RGB")

# h'face
device = 'cuda'
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

inputs = processor(image, input_points=input_points, input_labels=input_labels,return_tensors="pt").to(device)
inputs['multimask_output'] = True
with torch.no_grad():
    outputs = model(**inputs)
hface_masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
hface_masks = np.array([h.numpy() for h in hface_masks]).squeeze().astype('uint8')
hface_confidence_idx = outputs['iou_scores'][0][0].cpu().numpy()  # Index of the most confident mask

# meta 
sam = sam_model_registry["vit_b"](checkpoint="/media/jbishop/WD4/brainmets/sam_models/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
predictor.set_image(np.array(image))
meta_masks, meta_confidence_idx, _ = predictor.predict(point_coords=np.array(input_points).squeeze(0), point_labels=np.array(input_labels).squeeze(0), multimask_output=True)
meta_masks = meta_masks.astype('uint8')

# plot results
for m,c in zip([hface_masks,meta_masks],[hface_confidence_idx,meta_confidence_idx]):
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    mask_list = [m[0], m[1], m[2]]

    for i, mask in enumerate(mask_list, start=1):
        overlayed_image = np.array(image).copy()

        overlayed_image[:,:,0] = np.where(mask == 1, 255, overlayed_image[:,:,0])
        overlayed_image[:,:,1] = np.where(mask == 1, 0, overlayed_image[:,:,1])
        overlayed_image[:,:,2] = np.where(mask == 1, 0, overlayed_image[:,:,2])
        
        axes[i].imshow(overlayed_image)
        axes[i].set_title('Mask {}, iou={:.2f}'.format(i,c[i-1]))
    for ax in axes:
        ax.axis('off')

    plt.show()
    a=1


        
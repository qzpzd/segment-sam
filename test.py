# #################################################代码1#################################
# from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
# import cv2
# import os
# import numpy as np 
# import matplotlib.pyplot as plt

# image = cv2.imread('./1498378448034073286.jpg')
# save_path = "./"
# # predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth"))
# # predictor.set_image(image)

# mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="./sam_vit_h_4b8939.pth"))
# masks = mask_generator.generate(image)

# # masks, _, _ = predictor.predict()
# def show_anns(image,anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
   
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         img = np.ones((m.shape[0], m.shape[1], 3))
#         img = image.copy
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             img[:,:,i] = color_mask[i]
#         img = np.dstack((img, m*0.35))
#         #image[ann["bbox"]]=img
#     plt.imsave("i.jpg",img)
# show_anns(image,masks)
# exit()
########################################代码2########################################
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def write_masks_to_folder(masks,img):    
    for i, mask_data in enumerate(masks):
        
        mask = mask_data["segmentation"]
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for j in range(3):
            img[:,:,j] = color_mask[j]

        plt.imshow(np.dstack((img, mask*0.35)))
        filename = f"{i}.png"
        path = f"./output/segment_{img_name}"
        if not os.path.exists(path):os.makedirs(path)
        seg_path = os.path.join(path, filename)
        plt.axis('off')
        plt.savefig(seg_path,dpi=100)
        plt.clf()
        #cv2.imwrite(os.path.join(seg_path, filename), mask * 255)
        #cv2.imwrite(os.path.join(seg_path, filename), tack_img)
        #plt.imsave(os.path.join(seg_path, filename), tack_img)

    

#read-image
image = cv2.imread('./1498378448034073286.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_name = os.path.basename('./1498378448034073286.jpg')[:-4]

image2 = image.copy()

#set-model
sam_checkpoint = "./sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "default"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#generator1
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

#generator2
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
masks2 = mask_generator_2.generate(image2)

#show-image
def show(masks,image,img_name):

    print(len(masks))
    print(masks[0].keys())
   
    #anything-seg
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(f"segment_{img_name}.png",dpi=100)
    plt.show() 
    plt.close()
   

if __name__=="__main__":
    show(masks2,image2,img_name)
    write_masks_to_folder(masks2,image2)
# import onnxruntime as ort
import numpy as np
from PIL import Image
from utils.preprocess import preprocess
from utils.show_result import show_result,show_palette_result
from utils.split_img import Crop_img
import warnings
import torch
import torch.nn as nn
import mmcv
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot, show_cell_pyplot
import time
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from scipy import ndimage, io
import cv2
warnings.filterwarnings("ignore")
from scipy.spatial.distance import cdist
from pytorch_grad_cam import GradCAM, GradCAMElementWise
def find_nearest_pixel(unknown_pixel, known_pixels, labels):
    '''
    This function will find the coordinates and label of the closest known point to an unknown point

    return: The closest known point and its label.
    '''
    distances = cdist([unknown_pixel], known_pixels)
    nearest_index = np.argmin(distances)
    nearest_pixel = known_pixels[nearest_index]
    nearest_label = labels[nearest_index]
    return nearest_pixel, nearest_label

    
def img_read(path, device):
    img_ori = Image.open(path).convert("RGB")
    h,w = img_ori.size

    img_ori = img_ori.resize([768, 512])

    old_img = img_ori
    rgb_img = np.float32(np.array(img_ori)) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.to(device)

    return input_tensor, old_img, (h,w)


class SegmentationModelOutputWrapper(nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        self.backbone = model.backbone
        self.head = model.decode_head
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
    def forward(self, x):
        out = self.head(self.backbone(x))
        out = self.upsample(out)
        return out


class SemanticSegmentationTarget:
    def __init__(self, category, mask, device):
        self.category = category
        self.mask = torch.from_numpy(mask)
        # if torch.cuda.is_available():
        self.mask = self.mask.to(device)
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


def Inference_the_model(imgpath, thresholds = {"Chloroplast" : 0.55,
                                                "Mitochondrion" : 0.55,
                                                "Vacuole" : 0.5,
                                                "Nucleus" : 0.5}, config_file=None, checkpoint_file=None):


    # build the model from a config file and a checkpoint file
    device ='cuda:0' if torch.cuda.is_available() else 'cpu'

    input_tensor, old_img, ori_size = img_read(imgpath, device)

    model = init_model(config_file, checkpoint_file, device=device)
    if not torch.cuda.is_available():
        model = revert_sync_batchnorm(model)

    model = SegmentationModelOutputWrapper(model)
    output = model(input_tensor)

    normalized_masks = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
    sem_classes = [
        'background', 'Chloroplast', 'Mitochondrion', 'Vacuole', 'Nucleus'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    masks = {}
    final_mask = {}
    for index, name in enumerate(['Chloroplast', 'Mitochondrion', 'Vacuole', 'Nucleus']):
        car_category = sem_class_to_idx[name]
        car_mask = normalized_masks[0, :, :, :].argmax(axis=0)
        car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
        car_mask_float = np.float32(car_mask == car_category)
        mask = np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)
        masks[name] = mask

        # define the target layer, conv_seg is the fusion layer of OrgSegNet
        target_layers = [model.model.decode_head.conv_seg]
        targets = [SemanticSegmentationTarget(car_category, car_mask_float, device)]
        with GradCAM(model=model,
                    target_layers=target_layers,
                    use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets)[0, :]

        # The threshold is between 0 and 1.
        threshold = thresholds[name]

        split_mask = np.where(grayscale_cam<threshold, 0, 1)
        # fill the small hole 
        fill_image = ndimage.binary_fill_holes(split_mask, structure=np.ones((15,15))).astype(np.uint8)
        # label each instance mask
        labeled_image, num_labels = ndimage.label(fill_image)

        # Initialize a label image
        ori_mask = 255*np.array(car_mask_float).astype("uint8")
        labeled_mask = labeled_image
        pixel_add = ori_mask + labeled_mask

        # already know label： such as 1, 2, 3, 4, 5
        # we set background pixels as -255  
        # The unknow pixels we set as 0
        area_grow_image = pixel_add - 255

        w, h = area_grow_image.shape
        # We create an array to store the coordinate information of unknown points
        unknown_pixels_all = []
        for x in range(w):
            for y in range(h):
                if area_grow_image[x, y] == 0:
                    # 添加坐标
                    unknown_pixels_all.append(np.array([x, y]))


        # copy the original label_mask as the temp_mask
        # copy the original label_mask as the new_mask
        temp_mask = labeled_mask.copy()
        new_mask = labeled_mask.copy()
        for pix_index in range(len(unknown_pixels_all)):
            # Set the step size in the direction of eight connections
            # Find the nearest known point from eight directions
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            nearest_pixels = []
            nearest_labels = []
            temp_known_all = []
            temp_labels = []
            # Initialize a minimum number of steps (10000) to optimize the time consumption of the algorithm.
            # The shortest distance in each direction will be recorded.
            # If the algorithm walk more than this number of steps in a certain direction, that direction is ignored.
            minium_step = 10000
            for index, direction in enumerate(directions):
                # First find the nearest known point in each of the eight directions, 
                # while the loop stop until a known point in a certain direction is found

                # Initialize the unknow pixel coordinates (x, y) 
                x = unknown_pixels_all[pix_index][0]
                y = unknown_pixels_all[pix_index][1]
                step = 0

                total_step = 0
                while len(temp_known_all) == index:
                    step += 1
                    # check if the number of steps taken is less than the minimum number of steps
                    if step <= minium_step:
                        x = x + direction[0]
                        y = y + direction[1]

                        # Check if the coordinates of the point are out of bounds
                        if x<=0 or y<=0:
                            temp_known_all.append([0,0])
                            temp_labels.append(0)

                        # If the border has not been reached, continue searching for known points
                        elif 0 < x < 512 and 0 < y < 768:
                            current_pixel = np.array([x, y])
                            current_pixel_vaule = temp_mask[x, y]

                            if  current_pixel_vaule> 0:
                                temp_known_all.append(current_pixel)
                                temp_labels.append(current_pixel_vaule)
                                if step <= minium_step:
                                    minium_step = step

                        # Prevent strange errors from sending the algorithm into an infinite loop
                        else:
                            temp_known_all.append([0,0])
                            temp_labels.append(0)

                    # If the number of steps the algorithm takes in this direction exceeds the minimum number of steps, 
                    # stop the query in this direction
                    else:
                        temp_known_all.append([0,0])
                        temp_labels.append(0)
                        
            # Find the nearest known point and coordinates
            nearest_pixel, nearest_label = find_nearest_pixel(current_pixel, temp_known_all, temp_labels)
            nearest_pixels.append(nearest_pixel)
            nearest_labels.append(nearest_label)

            # Map labels of known points to unknown points
            new_mask[unknown_pixels_all[pix_index][0], unknown_pixels_all[pix_index][1]] = np.array(nearest_labels).max()

        # Perform an OPEN operation to eliminate possible noise
        kernel_size = (3, 3)  # init a kernel size
        kernel_shape = cv2.MORPH_RECT  #  init a kernel shape
        kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
        opened_image = cv2.morphologyEx(np.uint8(new_mask), cv2.MORPH_OPEN, kernel, iterations=1)

        final_mask[name] = opened_image
    image = show_result(normalized_masks.argmax(1), old_img)
    return old_img, masks, image, final_mask, ori_size




def inference_mmcv_model(model, img, result, palette, opacity=0.5):
    if hasattr(model, 'module'):
        model = model.module
    old_img = img
    if palette:
        palette = palette
    else:
        palette = [[255, 255, 255], [153, 221, 174], [173, 205, 14], [39, 137, 238], [150, 97, 244]]
    image = model.show_result(img, result, palette=palette, show=False, opacity=opacity)
    seg_img = show_palette_result(result)
    image = Image.blend(Image.fromarray(img), Image.fromarray(seg_img), 0.5)
    return old_img, seg_img, image



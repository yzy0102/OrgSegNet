import skimage
import cv2
from scipy.stats import norm
import numpy as np
from PIL import Image
import os
from utils.calculate_shape2 import VisGraphOther
import pandas as pd
def Cal_Area(Organelle_Instance, rule_per_pixel, ori_size):
    """
    Return the areas of the Orgfanelle instances
    """
    properties = skimage.measure.regionprops(Organelle_Instance)
    Areas = [properties[i].area for i in range(len(properties))]
    mag1 = (ori_size[0] / 768) * (ori_size[0] / 512)
    Areas = [round(cc * rule_per_pixel * rule_per_pixel * mag1, 4) for cc in Areas]
    
    return Areas

def Cal_intensity(Organelle_Instance, gray_img, background_intensity, organelle_path, ori_size, organelle, if_show = True):
    """
    Return the electron-intensity of the Orgfanelle instances
    """
    num_labels = Organelle_Instance.max()
    labeled_image = Organelle_Instance
    intensity = []
    separated_regions = []
    for label in range(1, num_labels + 1):
        mask = np.uint8(labeled_image == label)
        separated_region = cv2.bitwise_and(gray_img, gray_img, mask=mask)
        separated_regions.append(separated_region)

        hist = cv2.calcHist([np.array(separated_region).reshape([512, 768])], [0], mask[:, :], [256], [0, 256])
        x = np.array(hist)
        mu = np.mean(x)
        sigma = np.std(x)
        y = norm.pdf(hist, mu, sigma)
        intensity.append(y.argmin() / background_intensity)

    if if_show:
        for i, region in enumerate(separated_regions):
            # Image.fromarray(np.uint8(region)).convert("RGB").resize(ori_size).save(organelle_path +"/"+ str(i + 1)+ organelle + "_predict.jpg")
            Image.fromarray(np.uint8(region)).convert("RGB").resize(ori_size).save(os.path.join(organelle_path, str(i + 1)+ organelle + "_Instance.jpg"))
    return intensity


def Thread_func_get_morphological_parameter_of_organelle(root_path, organelle, Instance, background_intensity, gray_img, rule_per_pixel, ori_size):#):, old_img, h, w, user_path, rule_per_pixel):
    '''
    get the three morphological parameter of the organelle
    '''
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        
    organelle_path = os.path.join(root_path,  organelle)
    if not os.path.exists(organelle_path):
        os.mkdir(organelle_path)


    organelle_dict = {}
    number = []
    number = [i for i in range(1, Instance[organelle].max()+1)]
    area = Cal_Area(Instance[organelle], rule_per_pixel, ori_size)
    intensity = Cal_intensity(Instance[organelle], gray_img, background_intensity, organelle_path, ori_size, organelle)
    Complex = VisGraphOther(selectedImage = Instance[organelle], resolution = 5, outputFolder=organelle_path, organelle=organelle).sigmas
    organelle_dict["Serial Number"] = number
    organelle_dict["area/um2"] = area
    organelle_dict["electron-density"] = intensity
    organelle_dict["shape-complex"] = Complex

    df = pd.DataFrame(organelle_dict)
    df.to_csv(os.path.join(organelle_path, organelle + "_info" + ".csv"), index=None)
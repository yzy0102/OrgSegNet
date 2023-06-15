import numpy as np
import cv2

def ReturnSplitImg2(img, prop):


    palette = [[255, 255, 255], [174, 221, 153], [14, 205, 173], [238, 137, 39], [244, 97, 150]]
    # palette = [[0, 0, 0], [128, 0, 0], [0,128,0], [128, 128, 0], [0,0,128]]
    try:
        if len(img.dtype) == 0:
            pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        pixels = np.array(img)
    if prop == "Chloroplast":
        pixels[np.all(pixels != palette[1], axis=-1)] = (255, 255, 255)
    elif prop == "Mitochondrion":
        pixels[np.all(pixels != palette[2], axis=-1)] = (255, 255, 255)
    elif prop == "Vacuole":

        pixels[np.all(pixels != palette[3], axis=-1)] = (255, 255, 255)
    elif prop == "Nucleus":

        pixels[np.all(pixels != palette[4], axis=-1)] = (255, 255, 255)
    elif prop == "Back":
        pixels[np.all(pixels != palette[0], axis=-1)] = (255, 255, 255)

    else:
        print("[Back, Chloroplast, Mitochondrion, Vacuole, Nucleus]")
    return pixels

def ReturnSplitImg(img, prop):
    palette = [[0, 0, 0], [128, 0, 0], [0,128,0], [128, 128, 0], [0,0,128]]
    try:
        if len(img.dtype) == 0:
            pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        pixels = np.array(img)
    if prop == "Chloroplast":
        # pixels[np.all(pixels != palette[1], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[0], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[2], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[3], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[4], axis=-1)] = (255, 255, 255)

    elif prop == "Mitochondrion":
        pixels[np.all(pixels == palette[0], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[1], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[3], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[4], axis=-1)] = (255, 255, 255)
    elif prop == "Vacuole":
        pixels[np.all(pixels == palette[0], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[1], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[2], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[4], axis=-1)] = (255, 255, 255)
    elif prop == "Nucleus":
        pixels[np.all(pixels == palette[0], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[1], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[2], axis=-1)] = (255, 255, 255)
        pixels[np.all(pixels == palette[3], axis=-1)] = (255, 255, 255)
    elif prop == "Back":
        pixels[np.all(pixels != palette[0], axis=-1)] = (255, 255, 255)
    else:
        print("[Back, Chloroplast, Mitochondrion, Vacuole, Nucleus]")
    # pixels[np.all(pixels == palette[0], axis=-1)] = (255, 255, 255)
    return pixels


def calculate_result(result, prop):

    if prop == "Chloroplast":
        return np.where(result[0]==1, 1, 0)
    if prop == "Mitochondrion":
        return np.where(result[0]==2, 1, 0)
    if prop == "Vacuole":
        return np.where(result[0]==3, 1, 0)
    if prop == "Nucleus":
        return np.where(result[0]==4, 1, 0)

def Crop_img(old_img):

    old_img2 = np.array(old_img)
    w,h,_ = old_img2.shape
    old_img2 = cv2.cvtColor(old_img2, cv2.COLOR_RGB2GRAY)
    ret, old_img2 = cv2.threshold(old_img2, 254, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(old_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    area = []

    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    w2 = contours[max_idx][0][0][1]
    return old_img.crop([0, 0, h, w2])

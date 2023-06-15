import numpy as np
from PIL import Image
def show_result(result, old_img):

    result = result.reshape([512, 768])
    w, h = np.array(old_img).shape[0], np.array(old_img).shape[1]
    seg_img = np.zeros((512, 768, 3))
    palette = [[255, 255, 255], [174, 221, 153], [14, 205, 173], [238, 137, 39], [244, 97, 150]]
    colors = palette
    for c in range(0, 5):
        seg_img[:, :, 0] += ((result[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((result[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((result[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize([h, w], Image.ANTIALIAS)
    image = Image.blend(old_img, seg_img, 0.5)
    return image


def show_palette_result(result):
    seg = result[0]
    # palette = [[255, 255, 255], [174, 221, 153], [14, 205, 173], [238, 137, 39], [244, 97, 150]]
    palette = [[255, 255, 255], [153, 221, 174], [173, 205, 14], [39, 137, 238], [150, 97, 244]]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        seg_img[seg == label, :] = color
    # convert to BGR
    seg_img = seg_img[..., ::-1]
    return seg_img

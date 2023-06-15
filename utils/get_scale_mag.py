import pytesseract
from PIL import Image
def get_rule_pixel(img_path):
    '''
    Identify scale sizes and image magnifications in TEM images
    pytesseract should be installed first
    return: scale size, magnifications
    '''
    img = Image.open(img_path).crop().crop([3100, 3300, 3600, 3500])
    text = pytesseract.image_to_string(img ,lang="eng")

    with open(r"C:\LOG.txt", 'w+') as f:
        f.writelines(text)

    with open(r"C:\LOG.txt", 'r+') as f2:
        for line in f2.readlines():
            if line.strip("\n")[-2:] == 'um':
                rule = int(line.strip("\n").strip('um')) * 1000
                # print(rule)
            elif line.strip("\n")[-2:] == 'nm':
                rule = int(line.strip("\n").strip('nm'))
                # print(rule)
            elif line.strip("\n")[-1:] == 'x':
                mag = int(line.strip("\n").strip("Direct").strip('x').strip(" Mag:"))
                # print(mag)

    
    pixel = round(int(float(8.46666667e-05)*2000*2000), -1)
    print("pixel: ", pixel)
    return rule, pixel
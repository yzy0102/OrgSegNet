import numpy as np
import cv2
from scipy import signal
from scipy.stats import norm
def get_back_intensity(old_img):

    hist1 = cv2.calcHist([np.array(old_img)], [0], None, [256], [0, 256])

    hist1 = hist1[1:254]
    bins = np.arange(hist1.shape[0] + 1)
    x = np.array(hist1)
    mu = np.mean(x)
    sigma = np.std(x)
    y = norm.pdf(hist1, mu, sigma)

    xxx = bins[1:]
    yyy = y.ravel()
    z1 = np.polyfit(xxx, yyy, 100)
    p1 = np.poly1d(z1)
    yvals = p1(xxx)
    num_peak_3 = signal.find_peaks(yvals, distance=10)
    def get_tensity(num_peak_3=num_peak_3, y=y):
        num = [n - y.argmin() for n in num_peak_3[0]]
        num = np.array(num)
        num = np.where(num < 0, 255, num)
        return np.sort(num)[0] + y.argmin()
    return get_tensity(num_peak_3, y)





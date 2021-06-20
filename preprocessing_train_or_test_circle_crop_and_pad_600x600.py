import numpy as np
import cv2 as cv
import glob


def unsharp_mask(img, blur_size = (9,9), imgWeight = 1.5, gaussianWeight = -0.5):
    gaussian = cv.GaussianBlur(img, (5,5), 0)
    return cv.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)   

for opt in ["circle"]:
    for name in glob.glob("./train/"+opt+"/*"):
        img=cv.imread(name,cv.IMREAD_GRAYSCALE)
        
        print(img.shape)
        
        top=156
        bottom=155
        left=156
        right=156
        res=img[60:349, 27:315]
        result = cv.copyMakeBorder(res, top, bottom, left, right, cv.BORDER_REPLICATE)
        print(result.shape)
        
        img=result
        img = cv.blur(img, (5, 5))
        img = unsharp_mask(img)
        img = unsharp_mask(img)
        img = unsharp_mask(img)
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        img=cv.Canny(img,100,150)

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25,1))
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 25));
        detected_lines = cv.morphologyEx(img, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv.drawContours(img, [c], -1, (0,0,0), 2)
        detected_lines = cv.morphologyEx(img, cv.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv.findContours(detected_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv.drawContours(img, [c], -1, (0,0,0), 2)

        result=img
        cv.imwrite(name.replace("train","train1"), result)
        

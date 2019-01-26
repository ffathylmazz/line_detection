import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image): # step 3: canny function (keskin hatları daha belirgin gösterir)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)# step 1 : convert image to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # step 2 : reduce noise
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image): #step 4: region of interest(burası önemli:) öncelikle matploblib algoritması kullanarak şekli x ve y düzlemine oturtup sınırlarımızı belirledik)
    height = image.shape[0]
    polygons = np.array([[
        (200,height),(1100,height),(550, 250)]
    ])
    mask = np.zeros_like(image) #step 5: mask the interested region
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask) #step 6 : bitwise(canny fonksiyonun çıktısı ile mask image ı bitwise yapıyoruz)
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        if __name__ == '__main__':
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2),(255, 0, 0), 10)
    return  line_image

image =cv2.imread('10.png')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=5)
line_image = display_lines(lane_image,lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('result',combo_image)
cv2.waitKey(0)

#aaa
#hough transform technique (identify the lane line )
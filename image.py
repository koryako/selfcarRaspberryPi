# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # plt.imshow(mask)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

left_x1 = 0.
right_x1 = 0.
left_x2 = 0.
right_x2 = 0.
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    imshape = img.shape
    left_count = 0.
    right_count = 0.
    global left_x1
    global right_x1
    global left_x2
    global right_x2

    left_x1_sum = 0.
    left_x2_sum = 0.
    right_x1_sum = 0.
    right_x2_sum = 0.
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 != x1:
                slope = (y2 - y1)/(x2 - x1)
                if slope > 0.5:
                    right_count += 1
                    right_x1_sum += ((x2-x1)/(y2-y1))*(imshape[0]*0.6 - y1) + x1
                    right_x2_sum += ((x2-x1)/(y2-y1))*(imshape[0] - y1) + x1
                else:
                    if slope < -0.5:
                        left_count += 1
                        left_x1_sum += ((x2 - x1) / (y2 - y1)) * (imshape[0]*0.6 - y1) + x1
                        left_x2_sum += ((x2 - x1) / (y2 - y1)) * (imshape[0] - y1) + x1


    #for line in lines:
    #    for x1, y1, x2, y2 in line:
    #        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    if right_count != 0:
        right_x1 = right_x1_sum/right_count
        right_x2 = right_x2_sum/right_count
    cv2.line(img, (math.floor(right_x1), math.floor(imshape[0]*0.6)), ((math.floor(right_x2), imshape[0])), color, thickness)
    if left_count != 0:
        left_x1  = left_x1_sum/left_count
        left_x2  = left_x2_sum/left_count
    cv2.line(img, (math.floor(left_x1),  math.floor(imshape[0]*0.6)), ((math.floor(left_x2), imshape[0])),  color, thickness)
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., r=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, r)
def pipeline(image):
    image=mpimg.imread(image)
    gray=grayscale(image)
    kernel_size = 3 #5
    blur_gray = gaussian_blur(gray, kernel_size)
     # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 100
    edges = canny(blur_gray, low_threshold, high_threshold)
     # This time we are defining a four sided polygon to mask
    imshape = image.shape
    #print(imshape)
    vertices = np.array([[(0, imshape[0]), (imshape[1]*0.6, imshape[0]*0.45), (imshape[1]*0.6, imshape[0]*0.65),
                          (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
     # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 50  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    # Iterate over the output "lines" and draw lines on a blank image
    # line_image = np.copy(pl_original_image)*0 # creating a blank to draw lines on
    # draw_lines(line_image, lines, color=[255, 0, 0], thickness=2)
    lines_edges = weighted_img(line_image, image)
    
    return lines_edges
   

image=pipeline("img/solidWhiteRight.jpg")

print ("this image is:",type(image),'with dimesions:',image.shape)
plt.imshow(image,cmap='gray')

plt.show()
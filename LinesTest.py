
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import pygame


def add_points(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0] # Red
    thickness = -1
    radius = 15
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.circle(img2, (x0, y0), radius, color, thickness)
    cv2.circle(img2, (x1, y1), radius, color, thickness)
    cv2.circle(img2, (x2, y2), radius, color, thickness)
    cv2.circle(img2, (x3, y3), radius, color, thickness)
    return img2

def add_lines(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0] # Red
    thickness = 2
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.line(img2, (x0, y0), (x1, y1), color, thickness)
    cv2.line(img2, (x1, y1), (x2, y2), color, thickness)
    cv2.line(img2, (x2, y2), (x3, y3), color, thickness)
    cv2.line(img2, (x3, y3), (x0, y0), color, thickness)
    return img2




def warper(img):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def unwarp(img):

    # Compute and apply inverse perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)

    return unwarped



def filter_Use_WhiteYellowColors(image):
    """
    Filter the image, showing only a range of white and yellow
    """
    # Filter White
    threshold = 200 # white threshold
    high_threshold = np.array([255, 255, 255]) #Bright white
    low_threshold = np.array([threshold, threshold, threshold]) #Soft White
    wmask = cv2.inRange(image, low_threshold, high_threshold)
    white_img = cv2.bitwise_and(image, image, mask=wmask)

    # Filter Yellow
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Changing Color-space, HSV is better for object detection
    #hsv_img = np.copy(img)
    #For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
    high_threshold = np.array([110,255,255]) #Bright Yellow
    low_threshold = np.array([90,100,100]) #Soft Yellow
    ymask = cv2.inRange(hsv_img, low_threshold, high_threshold)
    yellow_img = cv2.bitwise_and(image, image, mask=ymask)

    # Combine the two above images
    filtered_img = cv2.addWeighted(white_img, 1., yellow_img, 1., 0.)

    # Convert to black/white
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
    binary_output2 = np.zeros_like(gray)
    binary_output2[(gray > 0)] = 1
    return binary_output2



def sobelEdgeDetection(img, sx=False, sy=False, thresh=(25, 200)):
    # Convert to HSV for simpler calculations
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if sx:
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))
    if sy:
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*sobely/np.max(sobely))
    else:
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
        sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
        #magnitude
        mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaled_sobel = np.uint8(255*sobely/np.max(sobely))

    # Create a binary mask where mag thresholds are me
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary



def cannyEdgeDetection(img, kernel_size=5, thresh=(25, 200)):

    #wrapedCanny = warper(img=img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply a Gaussian Blur
    gausImage = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Run the canny edge detection
    cannyImage = cv2.Canny(gray, thresh[0], thresh[1])
    cbinary = np.zeros_like(cannyImage)
    cbinary[(cannyImage > 0)] = 1

    return cbinary


def colorAndCanny(image, kernel_size=5, low_thresh=50, high_thresh=150):
    """
    Apply color filter and edge detection canny
    """
    # Filter White
    threshold = 200 # white threshold
    high_threshold = np.array([255, 255, 255]) #Bright white
    low_threshold = np.array([threshold, threshold, threshold]) #Soft White
    wmask = cv2.inRange(image, low_threshold, high_threshold)
    white_img = cv2.bitwise_and(image, image, mask=wmask)

    # Filter Yellow
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Changing Color-space, HSV is better for object detection
    #hsv_img = np.copy(img)
    #For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
    high_threshold = np.array([110,255,255]) #Bright Yellow
    low_threshold = np.array([90,100,100]) #Soft Yellow
    ymask = cv2.inRange(hsv_img, low_threshold, high_threshold)
    yellow_img = cv2.bitwise_and(image, image, mask=ymask)

    # Combine the two above images
    filtered_img = cv2.addWeighted(white_img, 1., yellow_img, 1., 0.)

    # Convert to black/white
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
    binary_output2 = np.zeros_like(gray)
    binary_output2[(gray > 0)] = 255

    # Apply a Gaussian Blur
    #gausImage = cv2.GaussianBlur(binary_output2, (kernel_size, kernel_size), 0)

    # Run the canny edge detection
    cannyImage = cv2.Canny(binary_output2, low_thresh, high_thresh)
    return cannyImage


def color_sodeOrCanny(color, sobelCanny):
    comb = np.zeros_like(sobelCanny)
    comb[(color == 1) | (sobelCanny == 1)] = 1

    return comb


def compare_images(image1, image2, image1_exp="Image 1", image2_exp="Image 2"):

    wrapedIm = warper(img=image2)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=30)
    ax2.imshow(image2, cmap='gray')
    ax2.set_title(image2_exp, fontsize=30)
    ax3.imshow(wrapedIm, cmap='gray')
    ax3.set_title("Wraped", fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def binaryPipline(img):
    whiteyellow = filter_Use_WhiteYellowColors(image=img)
    sobelImg = sobelEdgeDetection(img=img, thresh=(150, 225))
    colorSobel = color_sodeOrCanny(color=whiteyellow, sobelCanny=sobelImg)
    #compare_images(img, colorSobel, "Original", "ColorSobel")
    return colorSobel



ym_per_pix = 3*8/720 # meters per pixel in y dimension, 8 lines (5 spaces, 3 lines) at 10 ft each = 3m
xm_per_pix = 3.7/550 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters

def calc_line_fits(img):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

    #plt.figure()
    #plt.plot(histogram)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    left_fit = right_fit = left_fit_m = right_fit_m = np.array([0,0,0])
    # Step through the windows one by one
    for window in range(nwindows):
        #print(window)
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #print((win_xleft_low,win_y_low),(win_xleft_high,win_y_high))
        cv2.rectangle(img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low)
                          & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low)
                           & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    #print("Windows done!")
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    #print(lefty)
    #print(leftx)
    #print(righty)
    #print(rightx)


    if (len(lefty)>1) & (len(leftx)>1):
        left_fit = np.polyfit(lefty, leftx, 2)
        left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]

    if (len(righty)>1) & (len(rightx)>1):
        right_fit = np.polyfit(righty, rightx, 2)
        right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    return left_fit, right_fit, left_fit_m, right_fit_m, out_img



def applyLinesDetection(img, binpipeline, left_fit, right_fit):
    #left_fit = left_fit
    #right_fit = right_fit
    bimPipeWrap = warper(img=np.copy(binpipeline))

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bimPipeWrap).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, bimPipeWrap.shape[0]-1, bimPipeWrap.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result



def main(img):

    binpipeline=binaryPipline(img=img)

    img2 = warper(img=np.copy(binpipeline))
    left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(img=np.copy(img2))

    result = applyLinesDetection(img=img, binpipeline=binpipeline, left_fit=left_fit, right_fit=right_fit)
    #result = add_points(img=result, src=src)
    #result = add_lines(img=result, src=src)

    return result


### Settings
# Choose the number of sliding windows
global nwindows, margin, minpix, src, dst

h, w = 720, 1200

src = np.float32([
    [210, 700],
    [570, 460],
    [705, 460],
    [1075, 700]
])
# Points for the new image
dst = np.float32([
    [400, 720],
    [400, 0],
    [w-400, 0],
    [w-400, 720]
])



nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50




output = 'output_video/test_video_output123.mp4'
#videoPath = "D:\Python\ComputerVision\\roadVideo\\test2.mp4"
videoPath = "test_video/test_video.mp4"


pygame.display.set_caption('Test')
clip1 = VideoFileClip(videoPath)

clip1.reader.close()
clip1.audio.reader.close_proc()
clip = clip1.fl_image(main)

clip.preview()
#clip.write_videofile(output, audio=False)
pygame.quit()
os.system("taskkill /f /im ffmpeg-win64-v4.1.exe")
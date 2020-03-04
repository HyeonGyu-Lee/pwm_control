#!/usr/bin/env python

import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import time

def unwarp(img, src, dst):
    h,w = img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def abs_sobel_thresh(img, orient='x', thresh_min=10, thresh_max=255):
    gray = (cv2.cvtColor(img, cv2.COLOR_RGB2Lab))[:,:,0]
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y')
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    binary_output = sxbinary 
    return binary_output 

def hls_lthresh(img, thresh=(225, 255)): #200
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls_l = hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    binary_output = np.zeros_like(hls_l)
    binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
    
    return binary_output
'''
def canny_thresh(img, thresh=(10, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    canny = cv2.Canny(gray,10,255)

    scaled_canny = np.uint8(255*canny/np.max(canny))
    sxbinary = np.zeros_like(scaled_canny)
    sxbinary[(scaled_canny >= thresh[0]) & (scaled_canny <= thresh[1])] = 1
    binary_output = np.copy(sxbinary) 
    return binary_output
'''
def pipeline(img):
    img_sobelAbs = abs_sobel_thresh(img,'x',25)
    img_LThresh = hls_lthresh(img)
    #img_BThresh = lab_bthresh(img)
    #img_sobelMag = mag_thresh(img)
    #img_canny = canny_thresh(img)

    combined = np.zeros_like(img_LThresh)
    #combined = np.zeros_like(img_sobelAbs)
    #cv2.imshow('Lth', img_LThresh)
    #cv2.imshow('Bth', img_BThresh)
    #cv2.imshow('sobel', img_sobelAbs)
    #cv2.imshow('canny', img_canny)

    combined[(img_LThresh == 1) | (img_sobelAbs ==1)] = 1
    #combined[(img_LThresh == 1) | (img_sobelMag == 1)] = 1
    #combined[(img_LThresh == 1)] = 1
    #combined[(img_sobelAbs == 1)] = 1
    #combined[(img_canny == 1)] = 1
    
    return combined, Minv

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    

def sliding_window_polyfit(img, last_leftx_base, last_rightx_base, last_left_fit, check):
    histogram = np.sum(img[:,:], axis=0)

    origin_histogram = histogram
    
    # Check last bases was initialized
    if last_leftx_base != 0 or last_rightx_base != 0: 
        
        distrib_width = 200
        sigma = distrib_width / 12   
        
        leftx_range = np.arange(last_leftx_base - distrib_width/2, last_leftx_base + distrib_width/2, dtype=int)
        rightx_range = np.arange(last_rightx_base - distrib_width/2, last_rightx_base + distrib_width/2, dtype=int)

        weight_distrib = np.zeros(img.shape[1])
        for i in range(img.shape[1]):
            if i in leftx_range:
                weight_distrib[i] = gaussian(i, last_leftx_base, sigma)
            elif i in rightx_range:
                weight_distrib[i] = gaussian(i, last_rightx_base, sigma)

        histogram = np.multiply(histogram, weight_distrib)

    midpoint = np.int(histogram.shape[0]//2)
    quarter_point = np.int(midpoint//2)

    left_offset = 70
    right_offset1 = 70
    leftx_base = np.argmax(histogram[quarter_point-left_offset:quarter_point+left_offset]) + quarter_point - left_offset
    #print("leftx_base: {0}".format(leftx_base))

    rightx_base = np.argmax(histogram[midpoint+quarter_point-right_offset1:midpoint+quarter_point+right_offset1]) + midpoint+quarter_point-right_offset1
    

    nwindows = 10
    window_height = np.int(img.shape[0]/nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 50
    minpix = 400
    goodpix = 1100
    maxpix = 3100
    left_lane_inds = []
    right_lane_inds = []
    rectangle_data = []
    
    if check:
        count = 7
    else:
        count = 0   

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        
        
        win_xleft_low = leftx_current - margin 
        win_xleft_high = leftx_current + margin 
        
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        #print("point: {0}".format(good_left_inds))
        #print("point_number: {0}".format(len(good_left_inds)))
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if maxpix > len(good_left_inds) > goodpix:
                count += 1
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit, right_fit = (None, None)
    '''
    if len(leftx) != 0:
        if count > 4:
            left_fit = np.polyfit(lefty, leftx, 2)
            if count > 6:
                last_left_fit = left_fit
            print("count: {0}".format(count))
            count = 0
            
        else:
            left_fit = last_left_fit
            print("count: {0}".format(count))
            count = 0    
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2) 
    
    if len(leftx) != 0:
        if count > 4:
            left_fit = np.polyfit(lefty, leftx, 2)
            if count > 5:
                last_left_fit = left_fit
            print("count: {0}".format(count))
            count = 0
            
        else:
            left_fit = last_left_fit
            print("count: {0}".format(count))
            count = 0    
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    '''
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        print("left_fit: {0}".format(left_fit))
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2) 
    
    visualization_data = (rectangle_data, histogram, origin_histogram)
    
    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data, leftx_base, rightx_base, last_left_fit

def calc_curv_rad_and_center_dist(bin_img, l_fit, r_fit, l_lane_inds, r_lane_inds):
    ym_per_pix = 3.048/100 
    xm_per_pix = 3.7/378 
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    h = bin_img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
  
    
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx = nonzerox[l_lane_inds]
    lefty = nonzeroy[l_lane_inds] 
    rightx = nonzerox[r_lane_inds]
    righty = nonzeroy[r_lane_inds]
    
    if len(leftx) != 0 and len(rightx) != 0:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    
    if r_fit is not None and l_fit is not None:
        car_position = bin_img.shape[1]/2
        l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
        r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
        #print(center_dist)


    return left_curverad, right_curverad, center_dist

def draw_lane(original_img, binary_img, l_fit, r_fit, Minv):
    new_img = np.copy(original_img)
    if l_fit is None or r_fit is None:
        return original_img
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    h,w = binary_img.shape
    ploty = np.linspace(0, h-1, num=h)
    left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
    right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    #cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(0,255,255), thickness=25)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=25)

    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)

    return result

cap = cv2.VideoCapture('/home/jaes/Videos/testroad4.mp4')

if not cap.isOpened():
    print("ERROR: video not found")

# Initialization
last_leftx_base = 0
last_rightx_base = 0

margin = 80
rightx_first = 0
rightx_diff = 0 
last_left_fit = []
check = True

while(cap.isOpened()):
    # Start time
    start = time.time()

    ret, exampleImg = cap.read()

    exampleImg = cv2.resize(exampleImg, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
    exampleImg = cv2.cvtColor(exampleImg, cv2.COLOR_BGR2RGB)
        
    h,w = exampleImg.shape[:2]
    '''
    src = np.float32([(470 + rightx_diff,180),
                    (710 + rightx_diff,180), 
                    (100 + rightx_diff,630), 
                    (1080 + rightx_diff,630)])
    dst = np.float32([(250,0),
                    (w-250,0),
                    (250,h),
                    (w-250,h)])
    '''
    src = np.float32([(535 ,170),
                    (665,170), 
                    (260 ,700), 
                    (950 ,700)])
    dst = np.float32([(250,0),
                    (w-250,0),
                    (250,h),
                    (w-250,h)])
    
    exampleImg_unwarp, M, Minv = unwarp(exampleImg, src, dst)  
    #exampleImg_bin = cv2.Canny(exampleImg_unwarp, 10,70)
    exampleImg_bin, Minv = pipeline(exampleImg_unwarp)
    
    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data, last_leftx_base, last_rightx_base, last_left_fit = sliding_window_polyfit(exampleImg_bin, last_leftx_base, last_rightx_base, last_left_fit, check)
    
    check = False
    
    if rightx_first == 0:
        rightx_first = last_rightx_base
        
    rightx_diff = last_rightx_base - rightx_first
    rightx_diff = rightx_diff / 2.5
        
    #print("rightx_diff: {0}".format(rightx_diff))

    # End time
    end = time.time()

    rectangles = visualization_data[0]
    histogram = visualization_data[1]

    out_img = np.uint8(np.dstack((exampleImg_bin, exampleImg_bin, exampleImg_bin))*255)
    window_img = np.zeros_like(out_img)
    ploty = np.linspace(0, exampleImg_bin.shape[0]-1, exampleImg_bin.shape[0] )
    if not isinstance(left_fit, type(None)) and not isinstance(right_fit, type(None)):
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for rect in rectangles:
        cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
        cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
    nonzero = exampleImg_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 255, 255]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 255]
    
    
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
   
    rad_l, rad_r, d_center = calc_curv_rad_and_center_dist(exampleImg_bin, left_fit, right_fit, left_lane_inds, right_lane_inds)


    exampleImg_out1 = draw_lane(exampleImg, exampleImg_bin, left_fit, right_fit, Minv)
    
    # Time elapsed
    seconds = end - start

    fps = 1 / seconds

    print("Estimated frames per second : {0}".format(fps))
    cv2.imshow('unwarp', exampleImg_unwarp)
    cv2.imshow('canny', exampleImg_bin)
    cv2.imshow('result', result)
    cv2.imshow('binary', exampleImg)
    cv2.imshow('final', exampleImg_out1)
    #plt.plot(histogram)
    '''
    plt.imshow(exampleImg)
    x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
    y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
    plt.plot(x, y, color='#11ff11', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
    plt.show()
    '''    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()


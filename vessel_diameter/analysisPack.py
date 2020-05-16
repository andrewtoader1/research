'''
	@Author: Andrew Toader
	@Date: 23 April 2019
	@Purpose: Contains functions that are for use in the main program, vesselDiameters.py
'''

from nd2reader import Nd2 as reader
import math
import numpy as np
import cv2
import pickle
from numba import jit
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import copy
import os
import xlsxwriter as xl

'''

Contains the functions needed for analysis of Multiphoton data

To use: 'from analysisPack.py import *' at top of document

'''
##################################################
## save nd2 data to an np array
##################################################

#function that converts images to 
def map_uint16_to_uint8(img):

	lower_bound = np.min(img)
	upper_bound = np.max(img)

	out = np.concatenate([
			np.zeros(lower_bound, dtype = np.uint16),
			np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
			np.ones(2**16 - upper_bound, dtype = np.uint16) * 255
		])
	return out[img].astype(np.uint8)

#read the images from the nd2 format and convert files to uint8 datatype
def openND2(root_folder, fname):
	data = []
	with reader(root_folder + '/' + fname) as images:
		for i in range(len(images.z_levels)):
			#print(images[i])
			if(images[i] is None):
				continue
			eval('data.append(map_uint16_to_uint8(images[%d]))' % i)

	# remove any none types that may be there
	for i in range(len(data)):
		if(data[i] is None):
			data.remove(data[0])

	# Apply openCV3 gaussian filter
	for i in range(len(data)):
		data[i] = cv2.fastNlMeansDenoising(data[i], None, 10, 7, 21)
	return data

##################################################
## Export numpy array to a to pkl file
##################################################
def exportToPkl(images, root_folder, export_fname):
	with open(root_folder + '/' + export_fname, 'wb') as outfile:
		pickle.dump(images, outfile, pickle.HIGHEST_PROTOCOL)

##################################################
## open pkl file
##################################################
def openPkl(root_folder, fname):
	with open(root_folder + '/' + fname, 'rb') as infile:
		data = pickle.load(infile)
	return data

##################################################
## scroll through stack, returns z-level
##################################################
def scroll(data, frame1 = None):
	#create reference window (if given argument)
	if(np.any(frame1 != None)):
		cv2.namedWindow('Reference')
		cv2.moveWindow('Reference', 600, 0)
		cv2.imshow('Reference', frame1)

	i = 0
	while True:
		cv2.namedWindow('Scroll')
		cv2.moveWindow('Scroll', 0, 0)

		cv2.imshow('Scroll', data[i])

	    #scrolling system
		while True:
			key = cv2.waitKey(0) &0xFF
			if(key == ord('d') or key == ord('a') or key == ord('e')):
				break

		## RECORD LOCATION OF CLICK ###
		if(key == ord('e')):
			cv2.destroyAllWindows()
			return i
		elif (key == ord('d')): 
			i+=1
		elif (key == ord('a')): 
			i-=1

	    #dont let user go past range of array
		if(i > len(data) - 1): 
			i = len(data) - 1

		if (i < 0):
			i = 0

##################################################
## Create maximum Intensity Projection
##################################################
def mip(data, start, end):
	maxip = np.zeros_like(data[0], dtype='uint8')
	for i in range(len(data[0])):
		for j in range(len(data[0][0])):
			maximum = 0
			for k in range(start, end):
				if(data[k][i, j] > maximum):
					maximum = data[k][i, j]
			maxip[i, j] = maximum
	return maxip

##################################################
## threshold to create vessel mask
##################################################
def vesselMask(img):
	#copy image
	img_c = copy.deepcopy(img)
	# set initial threshold value
	thr = 50
	print('threshold = 50')

	# show original image in one window
	cv2.namedWindow('Original')
	cv2.moveWindow('Original', 0, 0)
	cv2.imshow('Original', img)

	while True:
		#copy image
		img_c = copy.deepcopy(img)

		#threshold the image
		thr_indx = img < thr
		thr_indx2 = img >= thr
		img_c[thr_indx] = 0
		img_c[thr_indx2] = 255

		cv2.namedWindow('Threshold')
		cv2.moveWindow('Threshold', 600, 0)
		cv2.imshow('Threshold', img_c)

		thr = input('threshold = ')

		if(thr == 'd'):
			break
		thr = int(thr)

	cv2.destroyAllWindows()
	return img_c


##################################################
## Align vessels (takes in 4 images)
##################################################

# helper functions to shift the image
def shift_left(img, l, r, t, b, ul, ll, ur, lr):
	l = np.insert(l, l.shape[1], img[:, 0], axis = 1)
	l = np.delete(l, 0, axis = 1)
	img = np.insert(img, img.shape[1], r[:, 0], axis = 1)
	img = np.delete(img, 0, axis = 1)
	r = np.insert(r, r.shape[1], 0, axis = 1)
	r = np.delete(r, 0, axis = 1)

	ul = np.insert(ul, ul.shape[1], t[:, 0], axis = 1)
	ul = np.delete(ul, 0, axis = 1)
	t = np.insert(t, t.shape[1], ur[:, 0], axis = 1)
	t = np.delete(t, 0, axis = 1)
	ur = np.insert(ur, ur.shape[1], 0, axis = 1)
	ur = np.delete(ur, 0, axis = 1)

	ll = np.insert(ll, ll.shape[1], b[:, 0], axis = 1)
	ll = np.delete(ll, 0, axis = 1)
	b = np.insert(b, b.shape[1], lr[:, 0], axis = 1)
	b = np.delete(b, 0, axis = 1)
	lr = np.insert(lr, lr.shape[1], 0, axis = 1)
	lr = np.delete(lr, 0, axis = 1)
	        
	return (img, l, r, t, b, ul, ll, ur, lr)

def shift_right(img, l, r, t, b, ul, ll, ur, lr):
	r = np.insert(r, 0, img[:, img.shape[1] - 1], axis = 1)
	r = np.delete(r, r.shape[1] - 1, axis = 1)
	img = np.insert(img, 0, l[:, l.shape[1] - 1], axis = 1)
	img = np.delete(img, img.shape[1] - 1, axis = 1)
	l = np.insert(l, 0, 0, axis = 1)
	l = np.delete(l, l.shape[1] - 1, axis = 1)

	ur = np.insert(ur, 0, t[:, t.shape[1] - 1], axis = 1)
	ur = np.delete(ur, ur.shape[1] - 1, axis = 1)
	t = np.insert(t, 0, ul[:, ul.shape[1] - 1], axis = 1)
	t = np.delete(t, t.shape[1] - 1, axis = 1)
	ul = np.insert(ul, 0, 0, axis = 1)
	ul = np.delete(ul, ul.shape[1] - 1, axis = 1)

	lr = np.insert(lr, 0, b[:, t.shape[1] - 1], axis = 1)
	lr = np.delete(lr, lr.shape[1] - 1, axis = 1)
	b = np.insert(b, 0, ll[:, ll.shape[1] - 1], axis = 1)
	b = np.delete(b, b.shape[1] - 1, axis = 1)
	ll = np.insert(ll, 0, 0, axis = 1)
	ll = np.delete(ll, ll.shape[1] - 1, axis = 1)

	return (img, l, r, t, b, ul, ll, ur, lr)

def shift_up(img, l, r, t, b, ul, ll, ur, lr):
	t = np.insert(t, t.shape[0], img[0, :], axis = 0)
	t = np.delete(t, 0, axis = 0)
	img = np.insert(img, img.shape[0], b[0, :], axis = 0)
	img = np.delete(img, 0, axis = 0)
	b = np.insert(b, b.shape[0], 0, axis = 0)
	b = np.delete(b, 0, axis = 0)

	ul = np.insert(ul, ul.shape[0], l[0, :], axis = 0)
	ul = np.delete(ul, 0, axis = 0)
	l = np.insert(l, l.shape[0], ll[0, :], axis = 0)
	l = np.delete(l, 0, axis = 0)
	ll = np.insert(ll, ll.shape[0], 0, axis = 0)
	ll = np.delete(ll, 0, axis = 0)

	ur = np.insert(ur, ur.shape[0], r[0, :], axis = 0)
	ur = np.delete(ur, 0, axis = 0)
	r = np.insert(r, r.shape[0], lr[0, :], axis = 0)
	r = np.delete(r, 0, axis = 0)
	lr = np.insert(lr, lr.shape[0], 0, axis = 0)
	lr = np.delete(lr, 0, axis = 0)

	return (img, l, r, t, b, ul, ll, ur, lr)



def shift_down(img, l, r, t, b, ul, ll, ur, lr):
	b = np.insert(b, 0, img[img.shape[0] - 1, :], axis = 0)
	b = np.delete(b, b.shape[0] - 1, axis = 0)
	img = np.insert(img, 0, t[t.shape[0] - 1, :], axis = 0)
	img = np.delete(img, img.shape[0] - 1, axis = 0)
	t = np.insert(t, 0, 0, axis = 0)
	t = np.delete(t, t.shape[0] - 1, axis = 0)

	ll = np.insert(ll, 0, l[l.shape[0] - 1, :], axis = 0)
	ll = np.delete(ll, ll.shape[0] - 1, axis = 0)
	l = np.insert(l, 0, ul[ul.shape[0] - 1, :], axis = 0)
	l = np.delete(l, l.shape[0] - 1, axis = 0)
	ul = np.insert(ul, 0, 0, axis = 0)
	ul = np.delete(ul, ul.shape[0] - 1, axis = 0)

	lr = np.insert(lr, 0, r[r.shape[0] - 1, :], axis = 0)
	lr = np.delete(lr, lr.shape[0] - 1, axis = 0)
	r = np.insert(r, 0, ur[ur.shape[0] - 1, :], axis = 0)
	r = np.delete(r, r.shape[0] - 1, axis = 0)
	ur = np.insert(ur, 0, 0, axis = 0)
	ur = np.delete(ur, ur.shape[0] - 1, axis = 0)

	return (img, l, r, t, b, ul, ll, ur, lr)

def c_shift_left(img, l, r, t, b, ul, ll, ur, lr):
	l = np.insert(l, l.shape[1], img[:, 0, :], axis = 1)
	l = np.delete(l, 0, axis = 1)
	img = np.insert(img, img.shape[1], r[:, 0, :], axis = 1)
	img = np.delete(img, 0, axis = 1)
	r = np.insert(r, r.shape[1], 0, axis = 1)
	r = np.delete(r, 0, axis = 1)

	ul = np.insert(ul, ul.shape[1], t[:, 0, :], axis = 1)
	ul = np.delete(ul, 0, axis = 1)
	t = np.insert(t, t.shape[1], ur[:, 0, :], axis = 1)
	t = np.delete(t, 0, axis = 1)
	ur = np.insert(ur, ur.shape[1], 0, axis = 1)
	ur = np.delete(ur, 0, axis = 1)

	ll = np.insert(ll, ll.shape[1], b[:, 0,  :], axis = 1)
	ll = np.delete(ll, 0, axis = 1)
	b = np.insert(b, b.shape[1], lr[:, 0, :], axis = 1)
	b = np.delete(b, 0, axis = 1)
	lr = np.insert(lr, lr.shape[1], 0, axis = 1)
	lr = np.delete(lr, 0, axis = 1)
	        
	return (img, l, r, t, b, ul, ll, ur, lr)

def c_shift_right(img, l, r, t, b, ul, ll, ur, lr):
	r = np.insert(r, 0, img[:, img.shape[1] - 1, :], axis = 1)
	r = np.delete(r, r.shape[1] - 1, axis = 1)
	img = np.insert(img, 0, l[:, l.shape[1] - 1, :], axis = 1)
	img = np.delete(img, img.shape[1] - 1, axis = 1)
	l = np.insert(l, 0, 0, axis = 1)
	l = np.delete(l, l.shape[1] - 1, axis = 1)

	ur = np.insert(ur, 0, t[:, t.shape[1] - 1, :], axis = 1)
	ur = np.delete(ur, ur.shape[1] - 1, axis = 1)
	t = np.insert(t, 0, ul[:, ul.shape[1] - 1, :], axis = 1)
	t = np.delete(t, t.shape[1] - 1, axis = 1)
	ul = np.insert(ul, 0, 0, axis = 1)
	ul = np.delete(ul, ul.shape[1] - 1, axis = 1)

	lr = np.insert(lr, 0, b[:, t.shape[1] - 1, :], axis = 1)
	lr = np.delete(lr, lr.shape[1] - 1, axis = 1)
	b = np.insert(b, 0, ll[:, ll.shape[1] - 1, :], axis = 1)
	b = np.delete(b, b.shape[1] - 1, axis = 1)
	ll = np.insert(ll, 0, 0, axis = 1)
	ll = np.delete(ll, ll.shape[1] - 1, axis = 1)
    
	return (img, l, r, t, b, ul, ll, ur, lr)

def c_shift_up(img, l, r, t, b, ul, ll, ur, lr):
	t = np.insert(t, t.shape[0], img[0, :, :], axis = 0)
	t = np.delete(t, 0, axis = 0)
	img = np.insert(img, img.shape[0], b[0, :, :], axis = 0)
	img = np.delete(img, 0, axis = 0)
	b = np.insert(b, b.shape[0], 0, axis = 0)
	b = np.delete(b, 0, axis = 0)

	ul = np.insert(ul, ul.shape[0], l[0, :, :], axis = 0)
	ul = np.delete(ul, 0, axis = 0)
	l = np.insert(l, l.shape[0], ll[0, :, :], axis = 0)
	l = np.delete(l, 0, axis = 0)
	ll = np.insert(ll, ll.shape[0], 0, axis = 0)
	ll = np.delete(ll, 0, axis = 0)

	ur = np.insert(ur, ur.shape[0], r[0, :, :], axis = 0)
	ur = np.delete(ur, 0, axis = 0)
	r = np.insert(r, r.shape[0], lr[0, :, :], axis = 0)
	r = np.delete(r, 0, axis = 0)
	lr = np.insert(lr, lr.shape[0], 0, axis = 0)
	lr = np.delete(lr, 0, axis = 0)

	return (img, l, r, t, b, ul, ll, ur, lr)



def c_shift_down(img, l, r, t, b, ul, ll, ur, lr):
	b = np.insert(b, 0, img[img.shape[0] - 1, :, :], axis = 0)
	b = np.delete(b, b.shape[0] - 1, axis = 0)
	img = np.insert(img, 0, t[t.shape[0] - 1, :, :], axis = 0)
	img = np.delete(img, img.shape[0] - 1, axis = 0)
	t = np.insert(t, 0, 0, axis = 0)
	t = np.delete(t, t.shape[0] - 1, axis = 0)

	ll = np.insert(ll, 0, l[l.shape[0] - 1, :, :], axis = 0)
	ll = np.delete(ll, ll.shape[0] - 1, axis = 0)
	l = np.insert(l, 0, ul[ul.shape[0] - 1, :, :], axis = 0)
	l = np.delete(l, l.shape[0] - 1, axis = 0)
	ul = np.insert(ul, 0, 0, axis = 0)
	ul = np.delete(ul, ul.shape[0] - 1, axis = 0)

	lr = np.insert(lr, 0, r[r.shape[0] - 1, :, :], axis = 0)
	lr = np.delete(lr, lr.shape[0] - 1, axis = 0)
	r = np.insert(r, 0, ur[ur.shape[0] - 1, :, :], axis = 0)
	r = np.delete(r, r.shape[0] - 1, axis = 0)
	ur = np.insert(ur, 0, 0, axis = 0)
	ur = np.delete(ur, ur.shape[0] - 1, axis = 0)

	return (img, l, r, t, b, ul, ll, ur, lr)

# helper function that performs alignment
def align(img, c_img, c_org_img):
	l = np.zeros_like(img)
	r = np.zeros_like(img)
	t = np.zeros_like(img)
	b = np.zeros_like(img)
	ul = np.zeros_like(img)
	ll = np.zeros_like(img)
	ur = np.zeros_like(img)
	lr = np.zeros_like(img)

	c_l = np.zeros_like(c_img)
	c_r = np.zeros_like(c_img)
	c_t = np.zeros_like(c_img)
	c_b = np.zeros_like(c_img)
	c_ul = np.zeros_like(c_img)
	c_ll = np.zeros_like(c_img)
	c_ur = np.zeros_like(c_img)
	c_lr = np.zeros_like(c_img)

	cv2.namedWindow('Shift')
	cv2.moveWindow('Shift', 0, 0)
	#scrolling system
	while True:
		#show colored image
		c_img.astype('uint8')
		cv2.imshow('Shift', c_img + c_org_img)

		while True:
			key = cv2.waitKey(0) &0xFF
			if(key == ord('d') or key == ord('a') or key == ord('w') or key == ord('z') or key == ord('t')): break

		if(key == ord('t')):
			cv2.destroyAllWindows()
			img.astype('uint8')
			return img

		elif (key == ord('d')): 
			(img, l, r, t, b, ul, ll, ur, lr) = shift_right(img, l, r, t, b, ul, ll, ur, lr)
			(c_img, c_l, c_r, c_t, c_b, c_ul, c_ll, c_ur, c_lr) = c_shift_right(c_img, c_l, c_r, c_t, c_b, c_ul, c_ll, c_ur, c_lr)
		elif (key == ord('a')): 
			(img, l, r, t, b, ul, ll, ur, lr) = shift_left(img, l, r, t, b, ul, ll, ur, lr)
			(c_img, c_l, c_r, c_t, c_b, c_ul, c_ll, c_ur, c_lr) = c_shift_left(c_img, c_l, c_r, c_t, c_b, c_ul, c_ll, c_ur, c_lr)
		elif (key == ord('w')): 
			(img, l, r, t, b, ul, ll, ur, lr) = shift_up(img, l, r, t, b, ul, ll, ur, lr)
			(c_img, c_l, c_r, c_t, c_b, c_ul, c_ll, c_ur, c_lr) = c_shift_up(c_img, c_l, c_r, c_t, c_b, c_ul, c_ll, c_ur, c_lr)
		elif (key == ord('z')): 
			(img, l, r, t, b, ul, ll, ur, lr) = shift_down(img, l, r, t, b, ul, ll, ur, lr)
			(c_img, c_l, c_r, c_t, c_b, c_ul, c_ll, c_ur, c_lr) = c_shift_down(c_img, c_l, c_r, c_t, c_b, c_ul, c_ll, c_ur, c_lr)



def vesselAlign(mip0, mip5, mip30, mip60, thr_0, thr_5, thr_30, thr_60):
	# create colored images to be able to differentiate
	c_0min = np.zeros((mip0.shape[0], mip0.shape[1], 3))
	c_0min.astype('uint8')
	c_5min = np.zeros((mip5.shape[0], mip5.shape[1], 3))
	c_5min.astype('uint8')
	c_30min = np.zeros((mip30.shape[0], mip30.shape[1], 3))
	c_30min.astype('uint8')
	c_60min = np.zeros((mip60.shape[0], mip60.shape[1], 3))
	c_60min.astype('uint8')

	c_0min[:, :, 0] = thr_0
	c_5min[:, :, 1] = thr_5
	c_30min[:, :, 2] = thr_30
	c_60min[:, :, 0] = thr_60
	c_60min[:, :, 2] = thr_60

	#shift the images
	mip5 = align(mip5, c_5min, c_0min)
	mip30 = align(mip30, c_30min, c_0min)
	mip60 = align(mip60, c_60min, c_0min)

	return (mip0, mip5, mip30, mip60)



##################################################
## Get locations of vessles
##################################################
def get_coordinates (event, x, y, flags, param):
	img = np.zeros([512, 512], dtype='uint8')

	global mouseX, mouseY
	if event == cv2.EVENT_LBUTTONDBLCLK:
		cv2.circle(img, (x, y), 1, (0, 0, 0), -1)
		mouseX, mouseY = x, y

#save reference image
def refImage(data, root_folder, location_array, vessel_name, time):
	#create results and reference_images directory if they dont exist
	if(not os.path.exists(root_folder + '/results')):
		os.makedirs(root_folder + '/results/reference_images')
	copy_data = copy.deepcopy(data)
	cv2.line(copy_data,(location_array[-1][1], location_array[-1][2]),(location_array[-2][1], location_array[-2][2]),(255,255,255), 3)
	cv2.putText(copy_data, '   ' + str(time) + '_' + vessel_name, (location_array[-1][1] - 4, location_array[-1][2] - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,255), 1)
	cv2.imwrite(root_folder + '/results/reference_images/' + str(time) + 'min_' + vessel_name + '.png', copy_data)

'''
CAUTION: OUTPUT IS IN X AND Y, TO MODIFY ARRAY, REVERSE X AND Y TO GET (r,c)
'''

#location_array list: (0: name of vessel; 1: x; 2: y; 3: frame # from data)
def get_vessel_reference(data, raw_data, root_folder, time = None, location_array0 = None, location_array5 = None, location_array30 = None, location_array60 = None):
	#final data array
	location_array =[]

	#keep track of names used
	name_list = []

	#looper var through images
	i = 0

	while True:
		cv2.namedWindow('MIP')
		cv2.moveWindow('MIP', 0, 0)
		cv2.setMouseCallback('MIP', get_coordinates)

		## Indicate past measured vessels with a line (white):
		if(location_array0 != None):
			for j in range(0, len(location_array0), 2):
				if(i == location_array0[j][3]):
					cv2.line(data[i],(location_array0[j][1], location_array0[j][2]),(location_array0[j+1][1], location_array0[j+1][2]),(255,255,255), 2)
					cv2.putText(data[i], '  0_' + location_array0[j][0], (location_array0[j][1] - 4, location_array0[j][2] - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,255), 1)

		if(location_array5 != None):
			for j in range(0, len(location_array5), 2):
				if(i == location_array5[j][3]):
					cv2.line(data[i],(location_array5[j][1], location_array5[j][2]),(location_array5[j+1][1], location_array5[j+1][2]),(255,255,255), 2)
					cv2.putText(data[i], '  5_' + location_array5[j][0], (location_array5[j][1] - 4, location_array5[j][2] - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,255), 1)

		if(location_array30 != None):
			for j in range(0, len(location_array30), 2):
				if(i == location_array30[j][3]):
					cv2.line(data[i],(location_array30[j][1], location_array30[j][2]),(location_array30[j+1][1], location_array30[j+1][2]),(255,255,255), 2)
					cv2.putText(data[i], '  30_' + location_array30[j][0], (location_array30[j][1] - 4, location_array30[j][2] - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,255), 1)

		if(location_array60 != None):
			for j in range(0, len(location_array60), 2):
				if(i == location_array60[j][3]):
					cv2.line(data[i],(location_array60[j][1], location_array60[j][2]),(location_array60[j+1][1], location_array60[j+1][2]),(255,255,255), 2)
					cv2.putText(data[i], '  30_' + location_array60[j][0], (location_array60[j][1] - 4, location_array60[j][2] - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, (255,255,255), 1)

		## Indicate already measured vessels with a line (gray):
		for j in range(0, len(location_array), 2):
			if(i == location_array[j][3]):
				cv2.line(data[i],(location_array[j][1], location_array[j][2]),(location_array[j+1][1], location_array[j+1][2]),(100,100,100), 2)
				cv2.putText(data[i], '  ' + location_array[j][0], (location_array[j][1] - 4, location_array[j][2] - 4), cv2.FONT_HERSHEY_PLAIN, 0.75, (100,100,100), 1)
		
		cv2.imshow('MIP', data[i])

	    #scrolling system
		while True:
			key = cv2.waitKey(0) &0xFF
			if(key == ord('q') or key == ord('e')):
				break

		## RECORD LOCATION OF CLICK ###
		if(key == ord('e')):

			#print the names of the vessles used
			print('--------------------------------------------')
			print('You have used: ')
			for j in range(0, len(location_array), 2):
				print(location_array[j][0])

			#ask user for name of new vesslel
			type_of = input('Type of vessel [USE ONLY: a#, v#, p#, pc#, sc#, tc#]\nOr type [c] if you accidentaly clicked to return to previous screen: ')
			
			#continue if user choses that his mouse click was accidental
			if(type_of == 'c'):
				print('Please chose next vessel.')
				print('--------------------------------------------')
				continue

			while(type_of in name_list):
				type_of = input('You have already used that name. Choose a different name: ')
			name_list.append(type_of)
			#append to data array
			location_array.append([type_of, mouseX, mouseY, i])
			#draw circle on screen
			cv2.circle(data[i],(location_array[-1][1],location_array[-1][2]), 4, (255, 255, 0), 1)
			i = location_array[-1][3]
		
			#measure endpoint
			print('Please Double-click Endpoint')
			cv2.namedWindow('MIP')
			cv2.moveWindow('MIP', 0, 0)
			cv2.setMouseCallback('MIP', get_coordinates)
			cv2.imshow('MIP', data[i])
			while True:
				key = cv2.waitKey(0) &0xFF
				if(key == ord('e')):
					location_array.append([type_of, mouseX, mouseY, i])

					#confirmation note and print to screen vessels measured so far
					print(type_of, 'Measured. Please chose next vessel.')
					print('--------------------------------------------')
					print('Vessels Measured:')
					for j in range(0, len(location_array), 2):
						print(location_array[j][0])
					#create reference imag
					refImage(raw_data[i], root_folder, location_array, type_of, time)

					break

		elif (key == ord('q')): 
			break

	#close window when done displaying
	cv2.destroyAllWindows()
	return location_array

##################################################
## Compute approximate slope per pixel (in row, col format)
##################################################
def calcSlope(r1, r2, c1, c2):
	if(c1 == c2):
		return 'UND'
	return (r2-r1)/(c2-c1)

##################################################
## Compute approximate slope per pixel
##################################################
def calcRInt(r, c, slope):
	return r - slope*c

##################################################
## find intensities of vessels and put in list
##################################################

#helper function for intensityList
def getIntensity(data, index, r1, r2, c1, c2, slope):
	intList = []

	flag = False
	#slope is UND, (FLAG set to escape from rest of cases)
	if(slope == 'UND'):
		flag = True
		#case 1
		if (r1 < r2):
			while (r1 != r2):
				r1 += 1
				intList.append(data[index][r1, c1]) 
		#case 2
		if (r1 > r2):
			while(r2 != r1):
				r2 += 1
				intList.append(data[index][r2,c2])

	if(flag == True): return intList

	#calc r-intercept
	b = calcRInt(r1, c1, slope)

	#slope <= 1 case
	if(abs(slope) <= 1):
		#case 1 and 2
		if(((r1 >= r2) and (c1 < c2)) or ((r1 <= r2) and (c1 < c2))):
			while (c1 != c2):
				c1 += 1
				ra1 = slope*c1 + b
				if((math.ceil(ra1) - ra1) > (ra1 - math.floor(ra1))):
					r1 = math.floor(ra1)
				else:
					r1 = math.ceil(ra1)
				intList.append(data[index][r1, c1])
		#case 2 and 3
		if(((r1 <= r2) and (c1 > c2)) or ((r1 >= r2) and (c1 > c2))):
			while(c2 != c1):
				c2 += 1
				ra2 = slope*c2 + b
				if((math.ceil(ra2) - ra2) > (ra2 - math.floor(ra2))):
					r2 = math.floor(ra2)
				else:
					r2 = math.ceil(ra2)
				intList.append(data[index][r2, c2])

	#slope > 1 case
	if(abs(slope) > 1):
		#case 1 and 2
		if(((r1 > r2) and (c1 < c2)) or ((r1 > r2) and (c1 > c2))):
			while(r2 != r1):
				r2 += 1
				ca2 = (r1 - b)/slope
				if((math.ceil(ca2) - ca2) > (ca2 - math.floor(ca2))):
					c2 = math.floor(ca2)
				else:
					c2 = math.ceil(ca2)
				intList.append(data[index][r2, c2])
		#case 2 and 3
		if(((r1 < r2) and (c1 > c2)) or ((r1 < r2) and (c1 < c2))):
			while(r1 != r2):
				r1 += 1
				ca1 = (r1 - b)/slope
				if((math.ceil(ca1) - ca1) > (ca1 - math.floor(ca1))):
					c1 = math.floor(ca1)
				else:
					c1 = math.ceil(ca1)
				intList.append(data[index][r1, c1])
	return intList

##########################################################################
#get profile of axis to reduce error for slanted lines
def getProfile(data, index, r1, r2, c1, c2, slope):
	#data holding variables
	profList = []
	previous = 0

	flag = False
	#slope is UND, (FLAG set to escape from rest of cases)
	if(slope == 'UND'):
		flag = True
		#case 1
		if (r1 < r2):
			while (r1 != r2):
				r1 += 1
				profList.append(previous + 1)
				previous = profList[-1]
		#case 2
		if (r1 > r2):
			while(r2 != r1):
				r2 += 1
				profList.append(previous + 1)
				previous = profList[-1]

	if(flag == True): return profList

	#calc r-intercept
	b = calcRInt(r1, c1, slope)

	#slope <= 1 case
	if(abs(slope) <= 1):
		#case 1 and 2
		if(((r1 >= r2) and (c1 < c2)) or ((r1 <= r2) and (c1 < c2))):
			while (c1 != c2):
				c1 += 1
				ra1 = slope*c1 + b
				if((math.ceil(ra1) - ra1) > (ra1 - math.floor(ra1))):
					r1 = math.floor(ra1)
					profList.append(previous + 1)
					previous = profList[-1]
				else:
					r1 = math.ceil(ra1)
					profList.append(previous + math.sqrt(2))
					previous = profList[-1]

		#case 2 and 3
		if(((r1 <= r2) and (c1 > c2)) or ((r1 >= r2) and (c1 > c2))):
			while(c2 != c1):
				c2 += 1
				ra2 = slope*c2 + b
				if((math.ceil(ra2) - ra2) > (ra2 - math.floor(ra2))):
					r2 = math.floor(ra2)
					profList.append(previous + 1)
					previous = profList[-1]
				else:
					r2 = math.ceil(ra2)
					profList.append(previous + math.sqrt(2))
					previous = profList[-1]

	#slope > 1 case
	if(abs(slope) > 1):
		#case 1 and 2
		if(((r1 > r2) and (c1 < c2)) or ((r1 > r2) and (c1 > c2))):
			while(r2 != r1):
				r2 += 1
				ca2 = (r1 - b)/slope
				if((math.ceil(ca2) - ca2) > (ca2 - math.floor(ca2))):
					c2 = math.floor(ca2)
					profList.append(previous + 1)
					previous = profList[-1]
				else:
					c2 = math.ceil(ca2)
					profList.append(previous + math.sqrt(2))
					previous = profList[-1]

		#case 2 and 3
		if(((r1 < r2) and (c1 > c2)) or ((r1 < r2) and (c1 < c2))):
			while(r1 != r2):
				r1 += 1
				ca1 = (r1 - b)/slope
				if((math.ceil(ca1) - ca1) > (ca1 - math.floor(ca1))):
					c1 = math.floor(ca1)
					profList.append(previous + 1)
					previous = profList[-1]
				else:
					c1 = math.ceil(ca1)
					profList.append(previous + math.sqrt(2))
					previous = profList[-1]

	return profList

##########################################################################
# use getIntensity to input the data into a list // also input the profile list
#intensities(array: 0:vessel name; 1: intensity list)
def intensityList(data, location_array):
	intensities = []
	for i in range(0, len(location_array), 2):
		#switch from (x,y) system to (r,c) system
		r1 = location_array[i][2]
		r2 = location_array[i+1][2]
		c1 = location_array[i][1]
		c2 = location_array[i+1][1]
		
		slope = calcSlope(r1, r2, c1, c2)
		intensities.append([location_array[i][0], getIntensity(data, location_array[i][3], r1, r2, c1, c2, slope)])
	return intensities

#return list of sum of profiles
def profileList(data, location_array):
	profile = []
	for i in range(0, len(location_array), 2):
		#switch from (x,y) system to (r,c) system
		r1 = location_array[i][2]
		r2 = location_array[i+1][2]
		c1 = location_array[i][1]
		c2 = location_array[i+1][1]
		
		slope = calcSlope(r1, r2, c1, c2)
		profile.append([location_array[i][0], getProfile(data, location_array[i][3], r1, r2, c1, c2, slope)])

	return profile

##################################################
## Calculate vessel diameter
##################################################
#helper function for normalizeIntensities (convert to int value from uint8)
def toInt(array):
	for i in range(len(array)):
		array[i] = int(array[i])
	return array

#set normalize graph of intensities so that they range from a scale of 0-1
def normalizeIntensities(intensityList, location_array):
	for i in range(0, len(location_array), 2):
		if(len(intensityList[int(i/2)][1]) == 0): continue

		intensityList[int(i/2)][1] = toInt(intensityList[int(i/2)][1])

		inten1 = intensityList[int(i/2)][1][0]
		inten2 = intensityList[int(i/2)][1][-1]
		normal_slope = (inten2 - inten1)/(len(intensityList[int(i/2)][1]))

		for j in range(len(intensityList[int(i/2)][1])):
			intensityList[int(i/2)][1][j] -= (normal_slope*j + inten1)

	return intensityList

#create intensiuties on a scale from 0-1
def scaleIntensities(intensityList, location_array):
	for i in range(0, len(location_array), 2):
		if(len(intensityList[int(i/2)][1]) == 0): continue
		#make max value 1
		scale = max(intensityList[int(i/2)][1])

		for j in range(len(intensityList[int(i/2)][1])):
			if (scale > 0 or scale < 0):
				intensityList[int(i/2)][1][j] = intensityList[int(i/2)][1][j]/scale
			else:
				intensityList[int(i/2)][1][j] = intensityList[int(i/2)][1][j]/0.00000000001
	return intensityList

#interpolate intensities (find 0.5) helper funciton to findBoundaries //returns x value
def interIntensities(x1, y1, x2, y2, thresh):
	slope = (y2-y1)/(x2-x1)
	b = y1 - slope*x1
	return (thresh-b)/slope


#helper function for findBoundaries to plot the intensity profile
def plotProfile(profile, mid, root_folder, fname, vessel_name):
	plt.plot(profile)
	plt.axhline(y=mid, color='r', linestyle='--')
	plt.xlabel('List Index')
	plt.ylabel('Intensity')
	plt.savefig(root_folder + '/results/reference_images/' + fname + '_' + vessel_name)
	plt.close()

#find the boundaries (0.5 intensity)
def findBoundaries(inten, location_array, root_folder=None, fname=None):
	boundary_loc = []
	for i in range(0, len(location_array), 2):
		if(len(inten[int(i/2)][1]) == 0):
			boundary_loc.append([inten[int(i/2)][0], [None]])
			continue

		#midpoint of line drawn
		midpoint = int(len(inten[int(i/2)][1])/2)

		#threshold value
		half = (max(inten[int(i/2)][1]) + min(inten[int(i/2)][1]))/2

		#save a plot of the profile and midpoint line to results/reference_images
		if((root_folder is not None) and (fname is not None)):
			plotProfile(inten[int(i/2)][1], half, root_folder, fname, location_array[i][0])

		last_val = 0
		for j in range(midpoint, -1, -1):
			'''if(inten[int(i/2)][1][j] == half):
				boundary_loc.append([location_array[i][0], [j]])
				break'''
			if(inten[int(i/2)][1][j] < half):
				boundary_loc.append([location_array[i][0], [interIntensities(j, inten[int(i/2)][1][j], j + 1, last_val, half)]])
				break
			last_val = inten[int(i/2)][1][j]

		last_val = 0
		for j in range(midpoint + 1, len(inten[int(i/2)][1]), 1):
			'''if(inten[int(i/2)][1][j] == half):
				boundary_loc[int(i/2)][1].append(j)
				break'''
			if(inten[int(i/2)][1][j] < half):
				boundary_loc[int(i/2)][1].append(interIntensities(j, inten[int(i/2)][1][j], j - 1, last_val, half))
				break
			last_val = inten[int(i/2)][1][j]

		if(len(boundary_loc[int(i/2)][1]) < 2):
			boundary_loc[int(i/2)][1] = [None]
	return boundary_loc

'''def findBoundaries(inten, location_array):
	boundary_loc = []
	for i in range(0, len(location_array), 2):
		if(len(inten[int(i/2)][1]) == 0):
			boundary_loc.append([inten[int(i/2)][0], [None]])
			continue

		#threshold value
		half = (max(inten[int(i/2)][1]) + min(inten[int(i/2)][1]))/2

		last_val = 0
		for j in range(len(inten[int(i/2)][1]) - 1, -1, -1):
			if(inten[int(i/2)][1][j] >= half):
				boundary_loc.append([location_array[i][0], j])
				break
			if(inten[int(i/2)][1][j] > half):
				boundary_loc.append([location_array[i][0], [interIntensities(j, inten[int(i/2)][1][j], j + 1, last_val, half)]])
				break
			last_val = inten[int(i/2)][1][j]

		last_val = 0
		for j in range(len(inten[int(i/2)][1])):
			if(inten[int(i/2)][1][j] >= half):
				boundary_loc[int(i/2)].append(j)
				break
			if(inten[int(i/2)][1][j] > half):
				boundary_loc[int(i/2)][1].append(interIntensities(j, inten[int(i/2)][1][j], j - 1, last_val, half))
				break
			last_val = inten[int(i/2)][1][j]
		if(len(boundary_loc[int(i/2)][1]) < 2):
			boundary_loc[int(i/2)][1] = [None]

	return boundary_loc'''

#interpolate projection list data
def interpolateProjections(x1, y1, x2, y2, xNeeded):
	slope = (y2-y1)/(x2-x1)
	b = y1 - slope*x1
	return slope*xNeeded + b

#get diameter from two boundary values and profile list
def realDistance(boundaryList, projectionList, location_array):
	distanceList = []
	for i in range(0, len(location_array), 2):
		if(boundaryList[int(i/2)][1] == [None]):
			distanceList.append([location_array[i][0], None])
			continue
		exactStart = boundaryList[int(i/2)][1][0]
		exactEnd = boundaryList[int(i/2)][1][1]
		endval = interpolateProjections(math.floor(exactEnd), projectionList[int(i/2)][1][math.floor(exactEnd)], math.ceil(exactEnd), projectionList[int(i/2)][1][math.ceil(exactEnd)], exactEnd)
		startval = interpolateProjections(math.floor(exactStart), projectionList[int(i/2)][1][math.floor(exactStart)], math.ceil(exactStart), projectionList[int(i/2)][1][math.ceil(exactStart)], exactStart)
		distanceList.append([location_array[i][0], abs(endval - startval)])
	return distanceList

#modify scaled distance in array (NO OUTPUT) (scale is 1.03microm/pix for multiphoton)
def scaledDistance(distance, location_array, scale):
	for i in range(0, len(location_array), 2):
		if(distance[int(i/2)][1] == None): continue
		distance[int(i/2)][1] = distance[int(i/2)][1]*scale


##################################################
## Write data to text file
##################################################

#rounding method that ignores None type
def roundN(num, percision):
	if(num is not None):
		return round(num, percision)
	else:
		return num
def toTxt(diameters0, diameters5, diameters30, diameters60, root_folder):
	#get names of vessels
	name_list = []
	nl0 = []
	nl5 = []
	nl30 = []
	nl60 = []

	for i in range(len(diameters0)):
		name_list.append(diameters0[i][0])
		nl0.append(diameters0[i][0])

	for i in range(len(diameters5)):
		nl5.append(diameters5[i][0])
		if(diameters5[i][0] not in name_list):
			name_list.append(diameters5[i][0])

	for i in range(len(diameters30)):
		nl30.append(diameters30[i][0])
		if(diameters30[i][0] not in name_list):
			name_list.append(diameters30[i][0])

	for i in range(len(diameters60)):
		nl60.append(diameters60[i][0])
		if(diameters60[i][0] not in name_list):
			name_list.append(diameters60[i][0])

	diamList0 = []
	diamList5 = []
	diamList30 = []
	diamList60 = []

	for i in range(len(name_list)):
		for j in range(len(diameters0)):
			if(diameters0[j][0] == name_list[i]):
				diamList0.append(diameters0[j][1])
			if(name_list[i] not in nl0):
				diamList0.append(None)
	for i in range(len(name_list)):
		for j in range(len(diameters5)):
			if(diameters5[j][0] == name_list[i]):
				diamList5.append(diameters5[j][1])
			if(name_list[i] not in nl5):
				diamList5.append(None)
	for i in range(len(name_list)):
		for j in range(len(diameters30)):
			if(diameters30[j][0] == name_list[i]):
				diamList30.append(diameters30[j][1])
			if(name_list[i] not in nl30):
				diamList30.append(None)
	for i in range(len(name_list)):
		for j in range(len(diameters60)):
			if(diameters60[j][0] == name_list[i]):
				diamList60.append(diameters60[j][1])
			if(name_list[i] not in nl60):
				diamList60.append(None)

	file = open(root_folder + '/results/diameters.txt', 'w')
	file.write("Vessel Diameters\n")
	file.write("\t0min\t5min\t30min\t60min\n")
	#write diameters to txt file
	for i in range(len(name_list)):
		file.write(name_list[i] + '\t' + str(roundN(diamList0[i], 2)) + '\t' + str(roundN(diamList5[i], 2)) + '\t' + str(roundN(diamList30[i], 2)) + '\t' + str(roundN(diamList60[i], 2)) + '\n')
	file.close()

def toTxtChange(diameters0, diameters5, diameters30, diameters60, root_folder):
	name_list = []
	nl0 = []
	nl5 = []
	nl30 = []
	nl60 = []

	#get all vessels measured
	for i in range(len(diameters0)):
		name_list.append(diameters0[i][0])
		nl0.append(diameters0[i][0])

	for i in range(len(diameters5)):
		nl5.append(diameters5[i][0])
		if(diameters5[i][0] not in name_list):
			name_list.append(diameters5[i][0])

	for i in range(len(diameters30)):
		nl30.append(diameters30[i][0])
		if(diameters30[i][0] not in name_list):
			name_list.append(diameters30[i][0])

	for i in range(len(diameters60)):
		nl60.append(diameters60[i][0])
		if(diameters60[i][0] not in name_list):
			name_list.append(diameters60[i][0])

	#check that vessel is measured at all 4 time points
	for i in range(len(name_list)):
		if((name_list[i] not in nl0) or (name_list[i] not in nl5) or (name_list[i] not in nl30) or (name_list[i] not in nl60)):
			name_list.pop(i)

	diamList0 = []
	diamList5 = []
	diamList30 = []
	diamList60 = []

	for i in range(len(name_list)):
		for j in range(len(diameters0)):
			if(diameters0[j][0] == name_list[i]):
				diamList0.append(diameters0[j][1])

	for i in range(len(name_list)):
		for j in range(len(diameters5)):
			if(diameters5[j][0] == name_list[i]):
				diamList5.append(diameters5[j][1])

	for i in range(len(name_list)):
		for j in range(len(diameters30)):
			if(diameters30[j][0] == name_list[i]):
				diamList30.append(diameters30[j][1])

	for i in range(len(name_list)):
		for j in range(len(diameters60)):
			if(diameters60[j][0] == name_list[i]):
				diamList60.append(diameters60[j][1])

	for i in range(len(diamList0)):
		diamList5[i] = ((diamList5[i] - diamList0[i])/diamList0[i])*100
		diamList30[i] = ((diamList30[i] - diamList0[i])/diamList0[i])*100
		diamList60[i] = ((diamList60[i] - diamList0[i])/diamList0[i])*100

	file = open(root_folder + '/results/pchange_diameters.txt', 'w')
	file.write("Percent Change From 0min\n")
	file.write("\t0min\t5min\t30min\t60min\n")
	#write diameters to txt file
	for i in range(len(name_list)):
		file.write(name_list[i] + '\t' + str(0) + '\t' + str(roundN(diamList5[i], 2)) + '\t' + str(roundN(diamList30[i], 2)) + '\t' + str(roundN(diamList60[i], 2)) + '\n')
	file.close()


##################################################
## Write data to excel file
##################################################

def toExcel(diameters0, diameters5, diameters30, diameters60, root_folder):
	#get names of vessels
	name_list = []
	nl0 = []
	nl5 = []
	nl30 = []
	nl60 = []

	for i in range(len(diameters0)):
		name_list.append(diameters0[i][0])
		nl0.append(diameters0[i][0])

	for i in range(len(diameters5)):
		nl5.append(diameters5[i][0])
		if(diameters5[i][0] not in name_list):
			name_list.append(diameters5[i][0])

	for i in range(len(diameters30)):
		nl30.append(diameters30[i][0])
		if(diameters30[i][0] not in name_list):
			name_list.append(diameters30[i][0])

	for i in range(len(diameters60)):
		nl60.append(diameters60[i][0])
		if(diameters60[i][0] not in name_list):
			name_list.append(diameters60[i][0])

	diamList0 = []
	diamList5 = []
	diamList30 = []
	diamList60 = []

	for i in range(len(name_list)):
		for j in range(len(diameters0)):
			if(diameters0[j][0] == name_list[i]):
				diamList0.append(diameters0[j][1])
			if(name_list[i] not in nl0):
				diamList0.append(None)
	for i in range(len(name_list)):
		for j in range(len(diameters5)):
			if(diameters5[j][0] == name_list[i]):
				diamList5.append(diameters5[j][1])
			if(name_list[i] not in nl5):
				diamList5.append(None)
	for i in range(len(name_list)):
		for j in range(len(diameters30)):
			if(diameters30[j][0] == name_list[i]):
				diamList30.append(diameters30[j][1])
			if(name_list[i] not in nl30):
				diamList30.append(None)
	for i in range(len(name_list)):
		for j in range(len(diameters60)):
			if(diameters60[j][0] == name_list[i]):
				diamList60.append(diameters60[j][1])
			if(name_list[i] not in nl60):
				diamList60.append(None)

	#write into excel
	wb = xl.Workbook(root_folder + '/' + 'results/diameters.xlsx')
	ws = wb.add_worksheet()

	#write title
	ws.write('A1', 'Diameters')
	ws.write('B2', 'Baseline')
	ws.write('C2', '5min Post')
	ws.write('D2', '30min Post')
	ws.write('E2', '60min Post')

	#write vessel names and values
	final = 0. #space out 2 items that go on this sheet (track end of this section)
	for i in range(len(name_list)):
		ws.write(i + 2, 0, name_list[i])
		ws.write(i + 2, 1, roundN(diamList0[i], 2))
		ws.write(i + 2, 2, roundN(diamList5[i], 2))
		ws.write(i + 2, 3, roundN(diamList30[i], 2))
		ws.write(i + 2, 4, roundN(diamList60[i], 2))
		final = i + 2
	final += 3
	###############
	# write in change of vessel from baseline
	#check that vessel is measured at all 4 time points
	for i in range(len(name_list)):
		if((name_list[i] not in nl0) or (name_list[i] not in nl5) or (name_list[i] not in nl30) or (name_list[i] not in nl60)):
			name_list.pop(i)

	diamList0 = []
	diamList5 = []
	diamList30 = []
	diamList60 = []

	for i in range(len(name_list)):
		for j in range(len(diameters0)):
			if(diameters0[j][0] == name_list[i]):
				diamList0.append(diameters0[j][1])

	for i in range(len(name_list)):
		for j in range(len(diameters5)):
			if(diameters5[j][0] == name_list[i]):
				diamList5.append(diameters5[j][1])

	for i in range(len(name_list)):
		for j in range(len(diameters30)):
			if(diameters30[j][0] == name_list[i]):
				diamList30.append(diameters30[j][1])

	for i in range(len(name_list)):
		for j in range(len(diameters60)):
			if(diameters60[j][0] == name_list[i]):
				diamList60.append(diameters60[j][1])

	for i in range(len(diamList0)):
		diamList5[i] = ((diamList5[i] - diamList0[i])/diamList0[i])*100
		diamList30[i] = ((diamList30[i] - diamList0[i])/diamList0[i])*100
		diamList60[i] = ((diamList60[i] - diamList0[i])/diamList0[i])*100

	#write title
	ws.write(final, 0, 'Percent Change from Baseline')
	ws.write(final + 1, 1, 'Baseline')
	ws.write(final + 1, 2, '5min Post')
	ws.write(final + 1, 3, '30min Post')
	ws.write(final + 1, 4, '60min Post')

	for i in range(len(name_list)):
		ws.write(i + 2 + final, 0, name_list[i])
		ws.write(i + 2 + final, 1, roundN(diamList0[i], 2))
		ws.write(i + 2 + final, 2, roundN(diamList5[i], 2))
		ws.write(i + 2 + final, 3, roundN(diamList30[i], 2))
		ws.write(i + 2 + final, 4, roundN(diamList60[i], 2))

	#close workbook to save
	wb.close()













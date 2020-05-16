'''
	@Author: Andrew Toader
	@Date: 23 April 2019
	@Purpose: Terminal application which allows user to measure the diameter of blood vessels in the brain and how they change over time
'''

from tkinter import filedialog
from tkinter import *
import os
import shutil
import copy

##################################################################################################
##################################################################################################
## FILE GATHERING AND MANAGING
##################################################################################################
##################################################################################################
os.system('clear')
print('Version 2.0')
print('--------------------------------------------')
print('')

# set current directory
print('Please select the folder of the root directory')

#select directory
while True:
	root = Tk()
	# hide tk window
	root.withdraw()
	root.root_dir =  filedialog.askdirectory()
	root_dir = root.root_dir
	root.destroy()
	#silence warning on osx and close the file window by calling input to terminal
	os.system('clear')

	#check that user chose valid directory
	exist = os.path.isfile(root_dir + '/' + '0min.nd2')
	if(not exist):
		print('Please select a valid folder containing files 0min.nd2, 5min.nd2, 30min.nd2, and 60min.nd2')
		continue
	input('Folder sucessfuly selected. Press [enter] to continue ')

	#check if a results file exists to notify user that this file has been analyzed
	exist = os.path.exists(root_dir + '/dat/distance0.pkl')

	#flag that controls whether to start over analysis or alter data
	start_over = False

	#if data has not been analyzed, start over
	if(exist == False):
		start_over = True
		break

	if(exist == True):
		u_in = input('It seems like this folder has already been analyzed. Would you like to [r]e-analyze, [a]lter results, or [q]uit: ')
		while(u_in not in ['r', 'a', 'q']):
			u_in = input('   Please enter either [r], [a], or [q]')
		print('--------------------------------------------')
	if(u_in == 'r'):
		start_over = True
		break
	elif(u_in == 'a'):
		break
	else:
		3/0

##################################################################################################
#file that contains all analysis functions
from analysisPack import *

#check if nd2 has been opened and saved to pkl
exists = os.path.isfile(root_dir + '/' + '0min.pkl') and os.path.isfile(root_dir + '/' + '5min.pkl') and os.path.isfile(root_dir + '/' + '30min.pkl') and os.path.isfile(root_dir + '/' + '60min.pkl')

#load nd2
if(exists != True):
	print('Opening 0min...')
	data0 = openND2(root_dir, '0min.nd2')
	print('Opening 5min...')
	data5 = openND2(root_dir, '5min.nd2')
	print('Opening 30min...')
	data30 = openND2(root_dir, '30min.nd2')
	print('Opening 60min...')
	data60 = openND2(root_dir, '60min.nd2')
	print('--------------------------------------------')

	##################################################################################################
	# crop the begining of the stacks
	print('Click rocket icon\nUse keys [a] and [d] to scroll\nPress [e] to accept as first frame')
	print('Crop 0min')
	crop0 = scroll(data0)
	data0.reverse()
	for i in range(crop0):
		data0.pop()
	data0.reverse()

	print('Crop 5min')
	crop5 = scroll(data5, data0[0])
	data5.reverse()
	for i in range(crop5):
		data5.pop()
	data5.reverse()

	print('Crop 30min')
	crop30 = scroll(data30, data0[0])
	data30.reverse()
	for i in range(crop30):
		data30.pop()
	data30.reverse()

	print('Crop 60min')
	crop60 = scroll(data60, data0[0])
	data60.reverse()
	for i in range(crop60):
		data60.pop()
	data60.reverse()

	#crop end of stacks
	len_array = [len(data0), len(data5), len(data30), len(data60)]
	cropto = min(len_array)

	while(len(data0) != cropto):
		data0.pop()

	while(len(data5) != cropto):
		data5.pop()

	while(len(data30) != cropto):
		data30.pop()

	while(len(data60) != cropto):
		data60.pop()

	##################################################################################################
	#save all of the arrays to pkl file
	exportToPkl(data60, root_dir, '60min.pkl')
	exportToPkl(data30, root_dir, '30min.pkl')
	exportToPkl(data5, root_dir, '5min.pkl')
	exportToPkl(data0, root_dir, '0min.pkl')

	#clear screen for next instrutions
	os.system('clear')
	print('Files sucessfuly cropped and stored in *.pkl files')
	print('--------------------------------------------')

#open pkl file if it already exists
else:
	data0 = openPkl(root_dir, '0min.pkl')
	data5 = openPkl(root_dir, '5min.pkl')
	data30 = openPkl(root_dir, '30min.pkl')
	data60 = openPkl(root_dir, '60min.pkl')

	# signal to user that the files have been loaded
	os.system('clear')
	print('Files Sucessfuly Opened')
	print('--------------------------------------------')

##################################################################################################
##################################################################################################
## CREATE MIP
##################################################################################################
##################################################################################################
if(os.path.exists(root_dir + '/mips60.pkl') == False):
	print('')
	print('Select MIP')
	print('')
	print('Click rocket icon\nUse keys [a] and [d] to scroll\nPress [e] to accept as start of MIP\n--SELECT FIRST IMAGE (most of the time)')
	startFrame = scroll(data0)
	os.system('clear')
	print('Click rocket icon\nUse keys [a] and [d] to scroll\nPress [e] to accept as end of MIP\n--RIGHT BEFORE YOU START SEEING CAPILIARIES')
	endFrame = scroll(data0[startFrame:len(data0)])

	mip0 = mip(data0, startFrame, endFrame + startFrame)
	mip5 = mip(data5, startFrame, endFrame + startFrame)
	mip30 = mip(data30, startFrame, endFrame + startFrame)
	mip60 = mip(data60, startFrame, endFrame + startFrame)

	os.system('clear')
	print('')
	print('Threshold Image')
	print('')
	print('Click rocket icon\nType in value between 0 - 255\nPress [d] to accept as final thresholded image')

	thr_0 = vesselMask(mip0)
	thr_5 = vesselMask(mip5)
	thr_30 = vesselMask(mip30)
	thr_60 = vesselMask(mip60)

	# save thresholded image to pkl
	exportToPkl([thr_0], root_dir, 'thr_0.pkl')
	exportToPkl([thr_5], root_dir, 'thr_5.pkl')
	exportToPkl([thr_30], root_dir, 'thr_30.pkl')
	exportToPkl([thr_60], root_dir, 'thr_60.pkl')

	##
	# align the vessels
	##
	os.system('clear')
	print('')
	print('Align Images')
	print('')
	print('Click rocket icon\nUse keys [a], [d], [w], and [z] to move image\nPress [t] to accept as final image (WHEN THE 2 ARE ALIGNED)')

	(mip0, mip5, mip30, mip60) = vesselAlign(mip0, mip5, mip30, mip60, thr_0, thr_5, thr_30, thr_60)

	#save all final mips to pkl
	exportToPkl([mip0], root_dir, 'mips0.pkl')
	exportToPkl([mip5], root_dir, 'mips5.pkl')
	exportToPkl([mip30], root_dir, 'mips30.pkl')
	exportToPkl([mip60], root_dir, 'mips60.pkl')

	os.system('clear')

else:
	mip0 = openPkl(root_dir, 'mips0.pkl')
	mip0 = mip0[0]
	mip5 = openPkl(root_dir, 'mips5.pkl')
	mip5 = mip5[0]
	mip30 = openPkl(root_dir, 'mips30.pkl')
	mip30 = mip30[0]
	mip60 = openPkl(root_dir, 'mips60.pkl')
	mip60 = mip60[0]

##################################################################################################
##################################################################################################
## GATHER VESSEL DATA AND ANALYZE
##################################################################################################
##################################################################################################

# put mips in list so that they work with how the below functions were written
mip0 = [mip0]
mip5 = [mip5]
mip30 = [mip30]
mip60 = [mip60]

if(start_over == True):
	print('Instructions:')
	print('  Click on rocket icon in dock')
	print('  DOUBLE CLICK to set point, and press [e] to accept point')
	print('  Clik will not be recorded until [e] is pressed')
	print('  Press [q] when you are done collecting data in this frame')
	print('--------------------------------------------')
	print('')
	#chose diameter of vessles
	print('Select Vessels for 0min')
	mip_send0 = copy.deepcopy(mip0)
	location_array0 = get_vessel_reference(mip_send0, mip0, root_dir, 0)

	print('Select Vessels for 5min')
	mip_send5 = copy.deepcopy(mip5)
	location_array5 = get_vessel_reference(mip_send5, mip5, root_dir, 5, location_array0)

	print('Select Vessels for 30min')
	mip_send30 = copy.deepcopy(mip30)
	location_array30 = get_vessel_reference(mip_send30, mip30, root_dir, 30, location_array0, location_array5)

	print('Select Vessels for 60min')
	mip_send60 = copy.deepcopy(mip60)
	location_array60 = get_vessel_reference(mip_send60, mip60, root_dir, 60, location_array0, location_array5, location_array30)

	##################################################################################################
	os.system('clear')
	print('Vessels sucessfuly selected')
	print('Analyzing Data...')

	#find intensities
	inten0 = intensityList(mip0, location_array0)
	inten5 = intensityList(mip5, location_array5)
	inten30 = intensityList(mip30, location_array30)
	inten60 = intensityList(mip60, location_array60)

	#get profile (actual length of line drawn)
	profile0 = profileList(mip0, location_array0)
	profile5 = profileList(mip5, location_array5)
	profile30 = profileList(mip30, location_array30)
	profile60 = profileList(mip60, location_array60)

	#normalize intensities (same start and end point)
	normal0 = normalizeIntensities(inten0, location_array0)
	normal5 = normalizeIntensities(inten5, location_array5)
	normal30 = normalizeIntensities(inten30, location_array30)
	normal60 = normalizeIntensities(inten60, location_array60)

	#scale intensities (range between 0-1, ideally)
	scl0 = scaleIntensities(normal0, location_array0)
	scl5 = scaleIntensities(normal5, location_array5)
	scl30 = scaleIntensities(normal30, location_array30)
	scl60 = scaleIntensities(normal60, location_array60)

	#find boundaries of vessels
	bound0 = findBoundaries(scl0, location_array0, root_dir, 'profile_0min')
	bound5 = findBoundaries(scl5, location_array5, root_dir, 'profile_5min')
	bound30 = findBoundaries(scl30, location_array30, root_dir, 'profile_30min')
	bound60 = findBoundaries(scl60, location_array60, root_dir, 'profile_60min')

	#find the distance between the two boundaries
	distance0 = realDistance(bound0, profile0, location_array0)
	distance5 = realDistance(bound5, profile5, location_array5)
	distance30 = realDistance(bound30, profile30, location_array30)
	distance60 = realDistance(bound60, profile60, location_array60)

	#find the actual distance (with scale)
	scaledDistance(distance0, location_array0, 1.03)
	scaledDistance(distance5, location_array5, 1.03)
	scaledDistance(distance30, location_array30, 1.03)
	scaledDistance(distance60, location_array60, 1.03)

##################################################################################################
#import dat if user selected to alter data only
if(start_over == False):
	location_array0 = openPkl(root_dir, 'dat/loc0.pkl')
	location_array5 = openPkl(root_dir, 'dat/loc5.pkl')
	location_array30 = openPkl(root_dir, 'dat/loc30.pkl')
	location_array60 = openPkl(root_dir, 'dat/loc60.pkl')
	distance0 = openPkl(root_dir, 'dat/distance0.pkl')
	distance5 = openPkl(root_dir, 'dat/distance5.pkl')
	distance30 = openPkl(root_dir, 'dat/distance30.pkl')
	distance60 = openPkl(root_dir, 'dat/distance60.pkl')
	mip0 = openPkl(root_dir, 'mips0.pkl')
	mip_send0 = copy.deepcopy(mip0)
	mip5 = openPkl(root_dir, 'mips5.pkl')
	mip_send5 = copy.deepcopy(mip5)
	mip30 = openPkl(root_dir, 'mips30.pkl')
	mip_send30 = copy.deepcopy(mip30)
	mip60 = openPkl(root_dir, 'mips60.pkl')
	mip_send60 = copy.deepcopy(mip60)

# Display resutls and ask user if they wish to change entries

#flag to enter while loop for changing data
while(True):
	print('--------------------------------------------')
	os.system('clear')
	print('Results:')

	#get names of all arteries used
	name_list = []
	for i in range(len(distance0)):
		name_list.append(distance0[i][0])

	for i in range(len(distance5)):
		if(distance5[i][0] not in name_list):
			name_list.append(distance5[i][0])

	for i in range(len(distance30)):
		if(distance30[i][0] not in name_list):
			name_list.append(distance30[i][0])

	for i in range(len(distance60)):
		if(distance60[i][0] not in name_list):
			name_list.append(distance60[i][0])

	#PRINT DATA TO SCREEN
	print('   0min')
	for i in range(len(name_list)):
		for j in range(len(distance0)):
			if(name_list[i] == distance0[j][0]):
				print('    ', name_list[i], ' - ', distance0[j][1])

	print('   5min')
	for i in range(len(name_list)):
		for j in range(len(distance5)):
			if(name_list[i] == distance5[j][0]):
				print('    ', name_list[i], ' - ', distance5[j][1])

	print('   30min')
	for i in range(len(name_list)):
		for j in range(len(distance30)):
			if(name_list[i] == distance30[j][0]):
				print('    ', name_list[i], ' - ', distance30[j][1])

	print('   60min')
	for i in range(len(name_list)):
		for j in range(len(distance60)):
			if(name_list[i] == distance60[j][0]):
				print('    ', name_list[i], ' - ', distance60[j][1])

	#change data
	u_in = input('Would you like to change data listed above? [y/n]: ')
	while((u_in != 'y') and (u_in != 'n')):
		u_in = input('    Would you like to change data listed above? [y/n]: ')

	#exit if no
	if(u_in == 'n'): break

	#change data
	change = input('Change [n]ame, change [d]ata entry (remeasure), [r]emove data point, or [a]dd data point?: ')
	while(change not in ['n', 'd', 'r', 'a']):
		change = input('    Invalid entry. Type either [n], [d], [r], or [a]: ')


	c_time = input('Time of data point to be changed/removed/added [0, 5, 30, 60, or \'q\' to cancel change]: ')
	while(c_time not in ['0', '5', '30', '60', 'q']):
		c_time = input('    Time of data point [0, 5, 30, 60, \'q\']: ')

	#quit option
	if(c_time == 'q'): continue

	#add list of available arteries
	c_time_list = []
	if(c_time == '0'):
		for i in range(len(distance0)): c_time_list.append(distance0[i][0])
	if(c_time == '5'):
		for i in range(len(distance5)): c_time_list.append(distance5[i][0])
	if(c_time == '30'):
		for i in range(len(distance30)): c_time_list.append(distance30[i][0])
	if(c_time == '60'):
		for i in range(len(distance60)): c_time_list.append(distance60[i][0])

	#remove measure if user choses
	if(change == 'r'):
		print('--------------------------------------------')
		for i in range(len(c_time_list)):
			print(c_time_list[i])
		print('--------------------------------------------')
		rm = input('Name you would like to remove (possible entries above): ')
		while(rm not in c_time_list):
			rm = input('    Invalid Entry (possible entries above): ')
		#get location of rm
		index = 0
		for i in range(len(c_time_list)):
			if(rm == c_time_list[i]):
				index = i

		if(c_time == '0'):
			distance0.pop(index)
			#index stays the same becasue list size changes after first pop
			location_array0.pop(index*2)
			location_array0.pop(index*2)
		if(c_time == '5'):
			distance5.pop(index)
			location_array5.pop(index*2)
			location_array5.pop(index*2)
		if(c_time == '30'):
			distance30.pop(index)
			location_array30.pop(index*2)
			location_array30.pop(index*2)
		if(c_time == '60'):
			distance60.pop(index)
			location_array60.pop(index*2)
			location_array60.pop(index*2)
		#skip data change(only selected to remove)
		continue
	#change name, if that is user's choice
	if(change == 'n'):
		for i in range(len(c_time_list)):
			print(c_time_list[i])
		old_name = input('Name you would like to change (possible entries above): ')
		while(old_name not in c_time_list):
			old_name = input('    Invalid Entry (possible entries above): ')
		#get location of old_name
		index = 0
		for i in range(len(c_time_list)):
			if(old_name == c_time_list[i]):
				index = i

		new_name = input('Change to: ')
		while(new_name in c_time_list):
			new_name = input('    You have already used that name. Enter new name: ')
		#assign new name
		if(c_time == '0'):
			distance0[index][0] = new_name
		if(c_time == '5'):
			distance5[index][0] = new_name
		if(c_time == '30'):
			distance30[index][0] = new_name
		if(c_time == '60'):
			distance60[index][0] = new_name
		#skip data change(only selected to chose name)
		continue

	## Add data point
	if(change == 'a'):
		if(c_time == '0'):
			print('--------------------------------------------')
			for i in range(len(c_time_list)):
				print(c_time_list[i])
			print('--------------------------------------------')
		if(c_time == '5'):
			print('--------------------------------------------')
			for i in range(len(c_time_list)):
				print(c_time_list[i])
			print('--------------------------------------------')
		if(c_time == '30'):
			print('--------------------------------------------')
			for i in range(len(c_time_list)):
				print(c_time_list[i])
			print('--------------------------------------------')
		if(c_time == '60'):
			print('--------------------------------------------')
			for i in range(len(c_time_list)):
				print(c_time_list[i])
			print('--------------------------------------------')
		add_vessel = input('Name of vessel to add (must not be in list above): ')
		while(add_vessel in c_time_list):
			add_vessel = input('    Invalid option. Re-enter: ')

		print('Instructions:')
		print('  Click on rocket icon in dock')
		print('  DOUBLE CLICK to set point, and press [e] to accept point')
		print('  Clik will not be recorded until [e] is pressed')
		print('  Press [q] when you are done chosing ONE vessel')
		print('--------------------------------------------')
		print('')

		#new vessel measure
		if(c_time == '0'):
			nlocation_array0 = get_vessel_reference(mip_send0, mip0, root_dir, 0, location_array0, location_array5, location_array30, location_array60)
			location_array0.append(nlocation_array0[0])
			location_array0.append(nlocation_array0[1])
			location_array0[-1][0] = add_vessel
			location_array0[-2][0] = add_vessel
		if(c_time == '5'):
			nlocation_array5 = get_vessel_reference(mip_send5, mip5, root_dir, 5, location_array0, location_array5, location_array30, location_array60)
			location_array5.append(nlocation_array5[0])
			location_array5.append(nlocation_array5[1])
			location_array5[-1][0] = add_vessel
			location_array5[-2][0] = add_vessel
		if(c_time == '30'):
			nlocation_array30 = get_vessel_reference(mip_send30, mip30, root_dir, 30, location_array0, location_array5, location_array30, location_array60)
			location_array30.append(nlocation_array30[0])
			location_array30.append(nlocation_array30[1])
			location_array30[-1][0] = add_vessel
			location_array30[-2][0] = add_vessel
		if(c_time == '60'):
			nlocation_array60 = get_vessel_reference(mip_send60, mip60, root_dir, 60, location_array0, location_array5, location_array30, location_array60)
			location_array60.append(nlocation_array60[0])
			location_array60.append(nlocation_array60[1])
			location_array60[-1][0] = add_vessel
			location_array60[-2][0] = add_vessel

		#find intensities
		if(c_time == '0'):
			ninten0 = intensityList(mip0, nlocation_array0)
			nprofile0 = profileList(mip0, nlocation_array0)
			nnormal0 = normalizeIntensities(ninten0, nlocation_array0)
			nscl0 = scaleIntensities(nnormal0, nlocation_array0)
			nbound0 = findBoundaries(nscl0, nlocation_array0,  root_dir, 'profile_0min')
			ndistance0 = realDistance(nbound0, nprofile0, nlocation_array0)
			scaledDistance(ndistance0, nlocation_array0, 1.03)
			distance0.append(ndistance0[0])
		if(c_time == '5'):
			ninten5 = intensityList(mip5, nlocation_array5)
			nprofile5 = profileList(mip5, nlocation_array5)
			nnormal5 = normalizeIntensities(ninten5, nlocation_array5)
			nscl5 = scaleIntensities(nnormal5, nlocation_array5)
			nbound5 = findBoundaries(nscl5, nlocation_array5,  root_dir, 'profile_5min')
			ndistance5 = realDistance(nbound5, nprofile5, nlocation_array5)
			scaledDistance(ndistance5, nlocation_array5, 1.03)
			distance5.append(ndistance5[0])
		if(c_time == '30'):
			ninten30 = intensityList(mip30, nlocation_array30)
			nprofile30 = profileList(mip30, nlocation_array30)
			nnormal30 = normalizeIntensities(ninten30, nlocation_array30)
			nscl30 = scaleIntensities(nnormal30, nlocation_array30)
			nbound30 = findBoundaries(nscl30, nlocation_array30,  root_dir, 'profile_30min')
			ndistance30 = realDistance(nbound30, nprofile30, nlocation_array30)
			scaledDistance(ndistance30, nlocation_array30, 1.03)
			distance30.append(ndistance30[0])
		if(c_time == '60'):
			ninten60 = intensityList(mip60, nlocation_array60)
			nprofile60 = profileList(mip60, nlocation_array60)
			nnormal60 = normalizeIntensities(ninten60, nlocation_array60)
			nscl60 = scaleIntensities(nnormal60, nlocation_array60)
			nbound60 = findBoundaries(nscl60, nlocation_array60,  root_dir, 'profile_60min')
			ndistance60 = realDistance(nbound60, nprofile60, nlocation_array60)
			scaledDistance(ndistance60, nlocation_array60, 1.03)
			distance60.append(ndistance60[0])

	### CHANGE DATA INPUT (REANALYZE)
	elif(change == 'd'):
		if(c_time == '0'):
			print('--------------------------------------------')
			for i in range(len(c_time_list)):
				print(c_time_list[i], '-', distance0[i][1])
			print('--------------------------------------------')
		if(c_time == '5'):
			print('--------------------------------------------')
			for i in range(len(c_time_list)):
				print(c_time_list[i], '-', distance5[i][1])
			print('--------------------------------------------')
		if(c_time == '30'):
			print('--------------------------------------------')
			for i in range(len(c_time_list)):
				print(c_time_list[i], '-', distance30[i][1])
			print('--------------------------------------------')
		if(c_time == '60'):
			print('--------------------------------------------')
			for i in range(len(c_time_list)):
				print(c_time_list[i], '-', distance60[i][1])
			print('--------------------------------------------')
		data_change = input('Which data point would you like to reanalyze? (enter name from list above): ')
		while(data_change not in c_time_list):
			data_change = input('    Invalid option. Re-enter: ')

		#display instructions
		print('Instructions:')
		print('  Click on rocket icon in dock')
		print('  DOUBLE CLICK to set point, and press [e] to accept point')
		print('  Clik will not be recorded until [e] is pressed')
		print('  Press [q] when you are done chosing ONE vessel')
		print('--------------------------------------------')
		print('')

		#get index to change
		index = 0
		for i in range(len(c_time_list)):
			if(data_change == c_time_list[i]):
				index = i

		#new vessel measure
		if(c_time == '0'):
			#remove that index from location array to update it
			location_array0.pop(index*2)
			#index stays the same becasue list size changes after first pop
			location_array0.pop(index*2)
			nlocation_array0 = get_vessel_reference(mip_send0, mip0, root_dir, 0, location_array0, location_array5, location_array30, location_array60)
			location_array0.append(nlocation_array0[0])
			location_array0.append(nlocation_array0[1])
		if(c_time == '5'):
			location_array5.pop(index*2)
			location_array5.pop(index*2)
			nlocation_array5 = get_vessel_reference(mip_send5, mip5, root_dir, 5, location_array0, location_array5, location_array30, location_array60)
			location_array5.append(nlocation_array5[0])
			location_array5.append(nlocation_array5[1])
		if(c_time == '30'):
			location_array30.pop(index*2)
			location_array30.pop(index*2)
			nlocation_array30 = get_vessel_reference(mip_send30, mip30, root_dir, 30, location_array0, location_array5, location_array30, location_array60)
			location_array30.append(nlocation_array30[0])
			location_array30.append(nlocation_array30[1])
		if(c_time == '60'):
			location_array60.pop(index*2)
			location_array60.pop(index*2)
			nlocation_array60 = get_vessel_reference(mip_send60, mip60, root_dir, 60, location_array0, location_array5, location_array30, location_array60)
			location_array60.append(nlocation_array60[0])
			location_array60.append(nlocation_array60[1])

		#find intensities
		if(c_time == '0'):
			ninten0 = intensityList(mip0, nlocation_array0)
			nprofile0 = profileList(mip0, nlocation_array0)
			nnormal0 = normalizeIntensities(ninten0, nlocation_array0)
			nscl0 = scaleIntensities(nnormal0, nlocation_array0)
			nbound0 = findBoundaries(nscl0, nlocation_array0,  root_dir, 'profile_0min')
			ndistance0 = realDistance(nbound0, nprofile0, nlocation_array0)
			scaledDistance(ndistance0, nlocation_array0, 1.03)
			distance0[index][1] = ndistance0[0][1]
		if(c_time == '5'):
			ninten5 = intensityList(mip5, nlocation_array5)
			nprofile5 = profileList(mip5, nlocation_array5)
			nnormal5 = normalizeIntensities(ninten5, nlocation_array5)
			nscl5 = scaleIntensities(nnormal5, nlocation_array5)
			nbound5 = findBoundaries(nscl5, nlocation_array5,  root_dir, 'profile_5min')
			ndistance5 = realDistance(nbound5, nprofile5, nlocation_array5)
			scaledDistance(ndistance5, nlocation_array5, 1.03)
			distance5[index][1] = ndistance5[0][1]
		if(c_time == '30'):
			ninten30 = intensityList(mip30, nlocation_array30)
			nprofile30 = profileList(mip30, nlocation_array30)
			nnormal30 = normalizeIntensities(ninten30, nlocation_array30)
			nscl30 = scaleIntensities(nnormal30, nlocation_array30)
			nbound30 = findBoundaries(nscl30, nlocation_array30,  root_dir, 'profile_30min')
			ndistance30 = realDistance(nbound30, nprofile30, nlocation_array30)
			scaledDistance(ndistance30, nlocation_array30, 1.03)
			distance30[index][1] = ndistance30[0][1]
		if(c_time == '60'):
			ninten60 = intensityList(mip60, nlocation_array60)
			nprofile60 = profileList(mip60, nlocation_array60)
			nnormal60 = normalizeIntensities(ninten60, nlocation_array60)
			nscl60 = scaleIntensities(nnormal60, nlocation_array60)
			nbound60 = findBoundaries(nscl60, nlocation_array60,  root_dir, 'profile_60min')
			ndistance60 = realDistance(nbound60, nprofile60, nlocation_array60)
			scaledDistance(ndistance60, nlocation_array60, 1.03)
			distance60[index][1] = ndistance60[0][1]
	print('Sucessfuly Updated')

# save all data to 'dat' folder
exist = os.path.exists(root_dir + '/dat')
if(not exist):
	os.mkdir(root_dir + '/dat')

#make directory to hold location array files (pkl format)
exportToPkl(location_array60, root_dir, '/dat/loc60.pkl')
exportToPkl(location_array30, root_dir, '/dat/loc30.pkl')
exportToPkl(location_array5, root_dir, '/dat/loc5.pkl')
exportToPkl(location_array0, root_dir, '/dat/loc0.pkl')

#export the final distance arrays to pkl
exportToPkl(distance60, root_dir, '/dat/distance60.pkl')
exportToPkl(distance30, root_dir, '/dat/distance30.pkl')
exportToPkl(distance5, root_dir, '/dat/distance5.pkl')
exportToPkl(distance0, root_dir, '/dat/distance0.pkl')

##################################################################################################
##################################################################################################
## EXPORT DATA TO TXT AND INTERPERT DATA
##################################################################################################
##################################################################################################

toTxt(distance0, distance5, distance30, distance60, root_dir)

#percent change
toTxtChange(distance0, distance5, distance30, distance60, root_dir)

toExcel(distance0, distance5, distance30, distance60, root_dir)













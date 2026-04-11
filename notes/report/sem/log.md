the sem images where taken from grid 8 on waver 2 
optical images are in the OM imaging folder 

### python scripts
#### processing_sem.py
	- main script
	- processes the edge analysis of the images by first converting them into their respective intensity profiles
		- that applies the gaussian and error func graph on it 
		from there the FWHM is converted to theta theside wall by using the h_px values ( which is the height of the Pd sample / nm_per_px of the image)
	- current issue is thtat the theta angle does not seem correct 

####  heatmap.py
	- vizulaizes teh .csvfile as a heatmap

#### scale_nm.py
	- opens the .tif image
	- allows you to click the scale bar lenght
	- converts the nm into nm_per_px
	- save into a .txt file to be used later in the processing.py



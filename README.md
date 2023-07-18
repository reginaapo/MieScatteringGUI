# Mie Scattering
## By: M. Regina A. Moreno

This is a GUI to facilitate Mie Scattering for spherical particles.

In the first column, you can select the experimental setup physical parameters that are relevant to the simulation. The automatic values correspond to the current AFN prototype at MIT.

In the second column, you can select the aerosol droplets' chemical composition from the selected list. You can also choose to add a SPHERICAL mineral particle to the aerosol droplet. 

The third column allows you to select the graphs you want to visualize. The last three graph options are especially relevant when comparing results to the data collected by the AFN.


### How to use the GUI:

Step 1: Select the desired experimental and aerosol physical parameters.

Step 2: Makes sure to press the "Apply" buttons to ensure your selected values are recorded.

Step 3: Select the graphs you wish to show.

Step 4: Press the "Run" button, and a new window should open and show the graphs you selected. 

Step 5: You can save the graphs you have produced as individual images by going back to the first window and pressing the new button "Save".




### Before getting started:
1. Download the "RI" folder and the "main.py" code, and ensure they are saved in the same folder.

2. Make sure you download the necessary python packages. Here is the list:

	- tkinker

	- random 

	- CTkMessagebox 

	- importlib.resources 

	- numpy 

	- matplotlib 

	- miepython 

	- scipy

	- os 



### Additional Information:

The documents in RI are the wavelength vs refractive index information for the different aerosols and minerals of interest. These were collected from publicly available data collected by previous authors. If you want to include any additional aerosols from the ones listed, please fill free to send me the link to the data and I'll be happy to add them.

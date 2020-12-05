# Code Folder

This folder contains all the preliminary preprocessing, data exploration, and modeling research for Materiall's image classification harness. Each subfolder in this directory organizes and outlines the process we took to reach our finalized model. Note - Many of these notebooks were used for individual exploration purposes and thus formal documentation may not have been included. However, our finalized code is stored in the 06_final_model folder and has all of the proper documentation needed to understand how to run and reproduce our completed work. 

To summarize: 
- **01_webscraping - 05_segmentation**: These folders contain scratch work and documents our thought process and exploration. 
- **06_final_model**: This folder contains our finalized code and has all the proper documentation. 


### Reproducing and Running our Model
The final notebook containing the model harness is located in 06_final_model within the 'server_nn.ipynb' notebook. A python script containing the same code can be found in 'server_nn.py'. 

To test our model within the 'server_nn.ipynb' Jupyter notebook, please follow these instructions:
1. Run all import and function cells
2. Run one of the following functions: 
	- ```python 
		run_test_harness('all')
	  ```
	- ```run_test_harness('binary')```
	- ```run_test_harness({insert individual dataset})``` - individual datasets available include 'fremont_dataset', 'sa_dataset', or 'ny_dataset'
	Additional parameters such as epochs, batch_size, shape, and other model hyperparameters are also optional declarations.

**Note**: The specificed datasets reference files and filepaths specific to Materiall's 54.186.23.133 server. 


Notebook titles are prefixed with the author's initials: 
- **BS**: Bharadwaj Swaminathan
- **DAD**: Daniel Alejandro del Carpio
- **PN**: Parker Nelson
- **ST**: Samantha Tang
- **VL**: Vincent Lao

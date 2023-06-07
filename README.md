# ArcheD
ArcheD - a novel residual neural network to predict amyloid CSF directly from amyloid PET scans.
# About the model
...

# How to use the model
1. Clone ArcheD repository.
2. **In the folder 'model_to_use'** unzip the file model.zip.
3. Run `pip install arched_package.zip`
4. Now you can run ArcheD model with your command line.

```  
arched [-h] [--output_name OUTPUT_NAME] path_to_directory folder_with_scans

 A novel residual neural network for predicting amyloid CSF directly from amyloid PET scans

 positional arguments:
  path_to_directory     the path to folder that **contains model (model_08-0.12_20_10_22.h5), arched_package.zip and folder with PET scans**, for ex. '~/abeta/ArcheD_run_example/'
  folder_with_scans     the name of the folder with scans or the path to it, for ex. 'scans'

 optional arguments:
  -h, --help            show this help message and exit
  --output_name OUTPUT_NAME, -o OUTPUT_NAME
                        name for the output file, for ex. 'arched_amyloid_csf_prediction'
```

Example of the command line

`arched '~/abeta/ArcheD_run_example/' 'scans' -o 'arched_amyloid_csf_prediction'` 

# Authors
...

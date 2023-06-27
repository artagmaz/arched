# ArcheD
ArcheD - a novel residual neural network to predict amyloid CSF directly from amyloid PET scans.
# About the model
ArcheD contains 35 layers and approximately 10.7 millions of parameters. 

Optimization: Adam optimization algorithm (Kingma & Ba, 2014) with an initial learning rate of 0.0001

Evaluation metrics: mean squared error (MSE)

Number of epochs: maximum of 150 epochs and stopped early, if loss in the validation dataset did not decrease for 15 epochs 

Batch size: 4

![plot](arched_architechture.png)


# How to use the model
1. Clone ArcheD repository from Github.
2. **In the folder 'model_to_use'** unzip the file model.zip.
3. Run `pip install arched_package.zip`
4. Now you can run ArcheD model with your command line.

```  
arched [-h] [--output_name OUTPUT_NAME] path_to_directory folder_with_scans

 A novel residual neural network for predicting amyloid CSF directly from amyloid PET scans

 positional arguments:
  path_to_directory     the path to folder that contains model (model_08-0.12_20_10_22.h5), arched_package.zip and folder with PET scans, for ex. '~/(your path)/model_to_use/'
  folder_with_scans     the name of the folder with scans (if the folder with scans is in path_to_directory) or the full path to it, for ex. 'scans' (as it locates in model_to_use folder) or '~/(your path)/scans'

 optional arguments:
  -h, --help            show this help message and exit
  --output_name OUTPUT_NAME, -o OUTPUT_NAME
                        name for the output file, for ex. 'arched_amyloid_csf_prediction'. Note: include the path if you want the output file to be saved not in the path_to_directory.
```

Example of the command line

`arched '~/model_to_use/' 'scans' -o 'arched_amyloid_csf_prediction'` 

5. If the model runs successfully, you will get the 'Model run successfully!' message and the CSV file will appear in your working directory. The file name will consist of the 'output_name', time and date of the model running.

# Authors
...

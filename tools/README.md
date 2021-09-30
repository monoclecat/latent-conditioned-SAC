## Running the tool to generate Csvs

Run the following command to get a anlayzing of the run experiment over the min, max, median and mean.
The results will be created in the base_folder in the folder toolOutput
```bash
python tools/generateCsvFromProgress.py --base_folder data/2021-07-30_exp_grid_walker2d-v2_disc0_cont2
```

## Plot generation script

If a direct plot generation is wished, the tool `tools/create_plots.py` can be used.
It must be run on the base `data` directory and can process all experiment results at once. 

```bash
python tools/create_plots.py --data-dir data
```

## Movement image generation script

With this script a movement image can be created. 

Example call:
``` bash
python tools/create_movement_image.py -bP data/serverExample/2021-07-28_exp_grid_hopper-v2_disc0_cont2/2021-07-28_12-36-18-exp_grid_hopper-v2_disc0_cont2_s0/images/ -irs 730 -ire 850 -iF 10
```

-bP - BasePath: gives the information where the images will be found and where the result will be safed
-irs - ImageRangeStart: gives the index or timestep of the startImage. On empty zero will be the default image
-ire - ImageRangeEnd: gives the index or timestep of the endImage. On empty the last image will be selected as the end
-iF - ImageFrequency: gives the imageFrequency of the recording image. (Just needed in case the timestep of the image should be used)

For taking the images please use the xml of the enviromentFolder and replace those with the gym xml files for mujoco.
The location of the gym xml file is in the python sideloadpackages of your enviroment.

In general the skybox should be white and the floor should be completly black so the horizon can be detected (The xml files will do that for the given enviroments).
Please have a look at the image in the folder "exampleImage" for the target image configuration.

## Create Pictogram for Skills

With this script a number of Movement Images can be created

Example call:
``` bash
python tools/create_movement_pictogram.py -rP data/serverExample/2021-07-28_exp_grid_hopper-v2_disc0_cont2/2021-07-28_12-36-18-exp_grid_hopper-v2_disc0_cont2_s0/ -nD 0 -nC 2 -t Hopper
```

-rP - RunPath: the path to the run base folder (the place where the images are placed on testing the policy and where testpolicy can be executed on)
-nD - Number of Discrete Skills
-nC - Number of Continuous Skills
-t - Enviromenttype: Only there to sort the images correctly in the folder
-s - Number of images that should be skipped (For the case the simulation is interrupted)

This script can only be used for the two configurations 2cont0disc and 1cont3disc.
If other configurations are needed, implement a permutation for this case to iterate through all skills (Or write a generic version :D).
Please create the folder for saving the images first to tools/MovementImages/{Enviromenttype}/disc{nD}_cont{nC}/.


## Questions?
In case there are any big questions, feel free to write an email to me: yannick.schaelow@student.kit.edu
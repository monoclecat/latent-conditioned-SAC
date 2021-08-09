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

-bP - BasePate: gives the information where the images will be found and where the result will be safed
-irs - ImageRangeStart: gives the index or timestep of the startImage. On empty zero will be the default image
-ire - ImageRangeEnd: gives the index or timestep of the endImage. On empty the last image will be selected as the end
-iF - ImageFrequency: gives the imageFrequency of the recording image. (Just needed in case the timestep of the image should be used)

For taking the images please use the xml of the enviromentFolder and replace those with the gym xml files for mujoco.
In general the skybox should be white and the floor should be completly black so the horizon can be detected and used
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

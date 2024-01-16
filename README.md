## Configuration
- Set the purpose, like turning on/off the debug mode, in the config file.
- Set some hard coded parameters in the config file, like the specific hyper-parameters for the model.
- Set the general parameters in the config file, like the path of the dataset.
- Check for your checkpoint demand in the config file. 
- Check the task type in the config file, like the self-supervised learning or the supervised learning.

### when you need resume from pretrained folder
- set EXPERIMENT.RESUME, EXPERIMENT.CHECKPOINT, EXPERIMENT.CHKP_IDX
  
### when you need debug the code
- set EXPERIMENT.DEBUG = True
- set EXPERIMENT.PROJECT = debug project

### when you need change the project object
- set EXPERIMENT.PROJECT = your project name, this will create a folder in the result folder
- set EXPERIMENT.NAME = your experiment name, this will change the folder name in the project folder
- set EXPERIMENT.TASK = your task name, this will change the trainer behavior

## Training Cmd

- Debug
```bash
PYTHONPATH=./ python scripts/train.py --cfg configs/debug/debug.yaml
```
- change cfg: EXPERIMENT>PROJECT. and name

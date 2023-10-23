## Configuration
- Set the purpose, like turning on/off the debug mode, in the config file.
- Set some hard coded parameters in the config file, like the specific hyper-parameters for the model.
- Set the general parameters in the config file, like the path of the dataset.
- Check for your checkpoint demand in the config file. 

## Training Cmd

- Debug
```bash
PYTHONPATH=./ python scripts/train.py --cfg configs/debug/debug.yaml
```

# General Framework for Deep Learning with PyTorch

## Requirements
  * Install python 3
  * Install pytorch == 1.1.0
  * Install tensorboardX

## Usage

### Define custom dataset

- define custom dataset class called `Gxxx`, you need to add a file called `data/Gxxx_dataset.py` and define a subclass `GxxxDataset` inherited from BaseDataset.
- You need to implement four functions:
```
<__init__>: initialize the class, first call BaseDataset.__init__(self, opt).
<__len__>:  return the size of dataset.
<__getitem__>: get a data point from data loader.
<modify_commandline_options>:add dataset-specific options and set default options.
```

Now you can use the dataset class by specifying flag '--dataset_mode Gxxx'.


### Define custom model

- To add a custom model class called `hh`, you need to add a file called `models/hh_model.py` and define a subclass `HhModel` inherited from BaseModel.
You need to implement the following five functions:
```
<__init__>: initialize the class; first call BaseModel.__init__(self, opt).
<set_input>: unpack data from dataset and apply preprocessing.
<forward>: produce intermediate results.
<optimize_parameters>: calculate loss, gradients, and update network weights.
<modify_commandline_options>: add model-specific options and set default options.
```
 - In the function `<__init__>`, you need to define three lists:
```
self.loss_names (str list):specify the training losses that you want to plot and save.
self.model_names (str list): define networks used in our training.
self.optimizers (optimizer list): define and initialize optimizers. 
```
Now you can use the model class by specifying flag '--model hh'.

### Example
You can create your custom dataset and model files following `example_dataset.py` and `example_model.py` 



## Training
Run `python train.py --verbose --model example   --dataset_mode example --name whatever_you_like`

check all the options in `opt/base_opts.py` and `opt/train_opts.py`

e.g. 
- --niter : which of iter to adjust learning rate
- --niter_decay : which of iter to linearly decay learning rate to zero
- --lr_policy : learning rate policy. [linear | step | plateau | cosine]
- --init_type : network initialization [normal | xavier | kaiming | orthogonal]
- --verbose : whether print the options
- ....


## Testing
Run `python test.py --model example   --dataset_mode example --name whatever_you_like --eval_epoch (int)`

## Visualization
`tensorboard --logdir='runs/' --port=8888`
  





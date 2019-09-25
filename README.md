# A Dynamic Approach to Accelerate Deep Learning Training
## Requirements
* Intel Caffe Framework (PyCaffe compilation)
* PIN Binary Analysis Tool 3.7
* Python 2.7
* An Intel processor with AVX512 support to use the PinTool
* Reduce Imagenet dataset. (200 categories, 256000 training images, 10000 validation images)

## Run Experiments
To run the experiments is important to have PIN 3.7 installed in order to compile the Pintool. Check the PIN guide to do this. Our pintool is in the pintool folder of this repository. Please after the compilation you will have a library called dynamic.so which do you will use to run the DNN models. Each experiments is on his own folder.

### Command
To run the command you will need to use the proper name of python script to execute the experiment. In our experiments we made runs each 3, 4 or 10 epochs depending on the CNN model under test.

```pin -inline 1 -t pintool_path/dynamic.so -- python trainAlexNetDynamic.py solver_alexnet_dynamic_ema_1000_batches.prototxt 100```

The file *'solver_alexnet_dynamic_ema_1000_batches.prototxt'* is the CNN model to use. In this case Alexnet. And the last parameter make reference to the parameter *numBatchesBF16=100xnumBatchesMP* referred in the paper.

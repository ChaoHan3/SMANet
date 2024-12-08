---
## Environment
- [PyTorch >= 1.8](https://pytorch.org/)
- [BasicSR >= 1.3.5](https://github.com/xinntao/BasicSR-examples/blob/master/README.md) 

**Dataset:**
 - https://www.digitalrocksportal.org
DeepRockSR-2D dataset covering 4000 500x500 high resolution images each of carbonate rocks and sandstones. The first 3200 images of each class are used as the training set, and the last 800 images are the test set. For x2 and x4 super-resolution reconstruction, we downsample the training images to 250x250 and 125x125 as inputs, respectively, to learn the mapping to the original resolution.
***Train***
During the training process, we randomly crop 16 images of size 64×64 and randomly perform horizontal flip and rotation from LR images as basic training inputs, using Adam optimiser to, for optimisation, starting learning rate set to 5 and adjusting the learning rate with cosine annealing during the training process, the network is trained for a total of 100. All experiments are performed on NVIDIA RTX 3080 GPU using the PyTorch framework.

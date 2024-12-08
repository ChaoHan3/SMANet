---
## Environment
- [PyTorch >= 1.8](https://pytorch.org/)
- [BasicSR >= 1.3.5](https://github.com/xinntao/BasicSR-examples/blob/master/README.md) 

**Dataset:**
 - https://www.digitalrocksportal.org
DeepRockSR-2D dataset covering 4000 500x500 high resolution images each of carbonate rocks and sandstones. The first 3200 images of each class are used as the training set, and the last 800 images are the test set. For x2 and x4 super-resolution reconstruction, we downsample the training images to 250x250 and 125x125 as inputs, respectively, to learn the mapping to the original resolution.

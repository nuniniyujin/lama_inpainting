# MVA Deep Learning project 2021

Team members :
* Gabriel BAKER : gabriel.baker@telecom-paris.fr
* Yujin CHO : yujin.cho@ens-paris-saclay.fr

## Improvement on Image inpainting with Fourier Convolutions

With the recent development of deep learning, excellent performance is being shown in the field of image inpainting, which includes tasks such as removing unnecessary parts from the image and naturally solving them or recovering certain partially damaged images. The purpose is to use the context present in the image to fill a masked region with plausible pixel values, capable of fooling a discriminator into saying the image was not modified.

In this work we implemented the recently presented network architecture LaMa[[arXiv]](https://arxiv.org/abs/2109.07161), which uses Fourier convolutions to inpaint images containing larges mask, while being robust to resolution. We also proposed two changes, evaluating them with the same metrics used in the original paper.

### Large Mask Inpainting

Roman et al. introduced LaMa [[arXiv]](https://arxiv.org/abs/2109.07161)(Large Mask Inpainting) model in 2021. It showed great result in image inpainting, by using Fourier convolutions. This innovative method of Fourier convolution was introdced in NeurIPS 2020 [[nips]](https://papers.nips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf) . It allows model to work on frequency domain so that we can have large receptive fields.
Receptive field is an important aspect in deep learning. It defines the region in the input data that produces the features.
Large receptive field helps to “understand“ large-scale structure such as patterns of natural images and to perform image synthesis.
For this there is two ways :
* Deep network : but problematic appear when propagating gradient until early layers
* Frequency domain : early layers have access to all image frequency synthesis

Using frequency domain allow early layers to have access full image information. 
And this information can be used to generate plausible image.

### Our experiments

* Changing the original loss function to add Wasserstein GAN Loss function
* Changing local convolution to spectral convolution
<center><img src="https://user-images.githubusercontent.com/80272042/152661363-e03d7248-3956-4ce8-98a8-1af3eb6d3b0e.png"  width="300" height="300"></center>

### Results
Result on test image with baseline model after 12 hours of training on 18250 images in colab notebook, from left to right:
![Result with baseline model](https://user-images.githubusercontent.com/80272042/152661798-2daf84ab-b502-436c-a774-ddbc327ca7c5.png)
(1) Original picture (2) Masked picture (3) Result on model trained only with square masks (4) Result on model trained with square masks and polygonal masks


Result on test image with multiples experiments after 12 hours of training on 18250 images in colab notebook, from left to right:
![result_eiffel](https://user-images.githubusercontent.com/80272042/152662097-dfe4a0e7-e42f-49f7-a307-4e2cc3a1836d.png)
(1) Original picture (2) Masked picture (3) baseline model (4) Model with spectral modification (5) Baseline model trained with Wasserstein loss

### Dataset

We used [Place365](http://places2.csail.mit.edu/download.html) dataset : Small images (256 * 256)

### Run training

```Python
!python main.py --data_path #Path/of/dataset
```

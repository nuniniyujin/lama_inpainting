# MVA Deep Learning project 2021

Team members :
* Gabriel BAKER : gabriel.baker@telecom-paris.fr
* Yujin CHO : yujin.cho@ens-paris-saclay.fr

## Improvement on Image inpainting with Fourier Convolutions

With the recent development of deep learning, excellent performance is being shown in the field of image inpainting, which includes tasks such as removing unnecessary parts from the image and naturally solving them or recovering certain partially damaged images. The purpose is to use the context present in the image to fill a masked region with plausible pixel values, capable of fooling a discriminator into saying the image was not modified.

In this work we implemented the recently presented network architecture [LaMa](https://arxiv.org/abs/2109.07161), which uses Fourier convolutions to inpaint images containing larges mask, while being robust to resolution. We also proposed two changes, evaluating them with the same metrics used in the original paper.

* Changing the original loss function to add Wasserstein GAN Loss function
* Changing local convolution to spectral convolution

![Screenshot](images/FFC_block_change.png)

### 

# Landsat7 Image Inpainting using Partial Convolutional U-net Architecture

**Objective :** On May 31, 2003, the Scan Line Corrector (SLC), which compensates for the forward motion of the Landsat-7 satellite, failed. Without an operating SLC, the sensor’s line of sight traces a zig-zag pattern along the satellite ground track.As a result, the imaged area is duplicated, with a width that increases toward the edge of the scene. The frequently used non-learning methods for filling these SLC-off Data gaps do not take the semantic information into account and the methods can only handle narrow holes. In this project, an attempt was made to propose a model for satellite image inpainting that operates robustly on SLC-off data gaps, and produces semantically meaningful predictions that incorporate smoothly with the rest of the image without the need for any additional post-processing or blending operation in order to have accurate data.

**Methods :** For this Image inpainting problem partial convolution layers with U-net like model architecture was evaluated. The resulting inpainting network could take 256×256 resolution greyscale NBRT Landsat 7 SLC-off satellite images taken over the Jamuna River as input and produce an inpainted image where there would be no data gaps. Weighted L1 as loss function, He normal initialization for weights and Adam as optimizer were used. 

**Results :**  

![alt text](https://github.com/antorhasan/landsat7_image_inpainting/blob/master/pngs/original1.png)

<p align="center">
  <b>Original Image with SLC-off Data Gap</b><br>
</p>

![alt text](https://github.com/antorhasan/landsat7_image_inpainting/blob/master/pngs/fixed1.png)

<p align="center">
  <b>Model Output</b><br>
</p>

<br/>
<br/>
<br/>

![alt text](https://github.com/antorhasan/landsat7_image_inpainting/blob/master/pngs/original2.png)

<p align="center">
  <b>Original Image with SLC-off Data Gap</b>
</p>

![alt text](https://github.com/antorhasan/landsat7_image_inpainting/blob/master/pngs/fixed2.png)

<p align="center">
  <b>Model Output</b><br>
</p>

</b>

**Conclusions :**
There haven’t been many works on satellite image inpainting. Most of the works focus on inpainting of general images or scenarios that are found in our everyday lives. The introduction of partial convolution layers and automatic mask update scheme implementation on satellite image data gaps inpainting problem was explored. The trained inpainting model could robustly handle holes of any shape, size, location, or distance from the image borders. Further, the performance did not deteriorate catastrophically as holes increased in size. 





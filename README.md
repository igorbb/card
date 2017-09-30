**Vehicle Detection Project**

The goals / steps of this project are the following:

* Train a car classifier.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report/input.png
[image2]: ./report/heatmap.png
[image3]: ./report/heatmaps.png
[image4]: ./report/frames.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Train a car classifier

Instead of going with HOG gradients, I wanted to try a new approach where I could leverage processing power of a GPU. I decided to use a neural network to tackle the problem of classification and avoid feature engineering while using convolution model. This is motivating for me, since I have previous experience with classical computer vision and object detection using histograms of oriented gradients, local binary patterns, sift, surf and other classical hand-engineered feature extraction techniques.

I am aware that there are network for object detection like [YOLO-darknet](https://pjreddie.com/darknet/yolo/) and [Single shot multi-box detection using mobilenets](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html). Nevertheless I want to see how far a simpler approach would take us. I also wanted to create a baseline where I can later modify the network to do both car and lade detection at the same time.

According to my last-project reviewer. It is okay to deviate from the proposed project pipeline, as long we have a motivating factor and we reach the same end-goal.


####1. Creating a Fast  reading dataset for training a neural network.

When training neural networks that uses images it is common to want to pack your data in a fast-reading format. This way one can avoid I/O over-head loading the data. For that I am using a wrapper library I co-developed to manage [LMDBs](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) datasets for machine learning. The library is open source and it is called [Pyxis](https://github.com/vicolab/ml-pyxis).


The idea is to create a dataset of images that shows the direct mapping from car-images to heat-maps.
To do that use the notebook named [Create LMDB.ipynb](./Create LMDB.ipynb). The notebook highlights the cells that are generating the heatmap from each image, and also identifies the cell blocks used to create the LMDB datasets

In the end we have a training and validation dataset (we skipped using a test set, which is a point to improove)
that has dashboard car images like this:

![alt text][image1]

and their associated car-heatmap:

![alt text][image2].


####2. Explain how you defined and train your classifier.


The idea is to using a convolution based image, that maps directly from input-image to heatmaps.
To do that I have opted to leverage ideas from convolution auto-encoders. We have a network that reduces the layers dimensions by using a convolutions and max pooling layers. After the information is compressed in to a smaller tensor, the network expands back to the original image input size (In this case an 180x320 image). Now the output as just one channel and it is trained with binary crossentropy to map to heatmaps


One can see at the networ defintion at cell [7] of the notebook [Model Train](./Model Train.ipynb).

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, None, None, 3)     0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 180, 320, 3)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 180, 320, 32)      13856     
_________________________________________________________________
batch_normalization_1 (Batch (None, 180, 320, 32)      128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 90, 160, 32)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 90, 160, 32)       147488    
_________________________________________________________________
batch_normalization_2 (Batch (None, 90, 160, 32)       128       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 45, 80, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 45, 80, 32)        147488    
_________________________________________________________________
batch_normalization_3 (Batch (None, 45, 80, 32)        128       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 23, 40, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 23, 40, 32)        147488    
_________________________________________________________________
batch_normalization_4 (Batch (None, 23, 40, 32)        128       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 12, 20, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 12, 20, 32)        147488    
_________________________________________________________________
batch_normalization_5 (Batch (None, 12, 20, 32)        128       
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 6, 10, 32)         0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 6, 10, 32)         147488    
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 12, 20, 32)        0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 12, 20, 32)        128       
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 12, 20, 32)        147488    
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 24, 40, 32)        0         
_________________________________________________________________
batch_normalization_7 (Batch (None, 24, 40, 32)        128       
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 24, 40, 32)        147488    
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 48, 80, 32)        0         
_________________________________________________________________
batch_normalization_8 (Batch (None, 48, 80, 32)        128       
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 48, 80, 64)        294976    
_________________________________________________________________
up_sampling2d_4 (UpSampling2 (None, 96, 160, 64)       0         
_________________________________________________________________
batch_normalization_9 (Batch (None, 96, 160, 64)       256       
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 98, 166, 32)       43040     
_________________________________________________________________
up_sampling2d_5 (UpSampling2 (None, 196, 332, 32)      0         
_________________________________________________________________
batch_normalization_10 (Batc (None, 196, 332, 32)      128       
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 180, 320, 1)       7073      
_________________________________________________________________
flatten_1 (Flatten)          (None, 57600)             0         
=================================================================
Total params: 1,392,769
Trainable params: 1,392,065
Non-trainable params: 704
________________________________________________________________
```

The input and lambda layers are there to make sure that every input image is resized to an 180x320 image.

From `conv2d_1` up to `conv2d_6` we are compressing the information by reducing the number of elements in the tensors.
From `conv2d_6` all the way up to output we are mapping the compressed information to a single channel heat-map.

Training is done in cell[8] of the same notebook.
Since we already are using a neural network that convolves the whole image with feature maps, this approach does not to rely and an extra set of sliding windows to do its classification.


#### What did you do to optimize the performance of your classifier?

Ultimately I have tried:
* Network using HOG features as inputs to map directly to heat-map
* Network using RGB images to map directly to heat-map
* Network using HSV images to map directly to heat-map
* Network using HSF images to map directly to heat-map
* Network using LAB images to map directly to heat-map

   The LAB images out-performed the other approaches; also there was no significant performance difference in making the LAB color space unit variance, or zero mean. Specially when using so many batch normalization inner layers.

* Classify smaller images with car/no-car, using RGB images

   This approach was following the pipeline propose in the project. I managed to classify 64x64 images with 98.9% accuracy. But the sliding window we need to get reliable detection made the solution to slow to the heatmap solution.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.

These images were generated using the notebook
[Heatmaps](./Heatmaps.ipynb).

![alt text][image3].


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Using the CNN I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In the following image you can see the orignal RGB image, followed by the heatmap and the selected bounding boxes:

![alt text][image4].


To weed out the false positive We have a running average of images. The threshold for car detection is computed over the running average. This is done in by the code in cell [4] in the notebook [Video](./Video.ipynb).

A better approach would be to actually use kalman filter to track the car position.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The proposed pipeline fails when we have car to far away from the camera. This is likely due to the filters on car-size we imposed when creating the heatmaps for the LMDBs.

The heatmaps are reliable and can be used to define ROI. A two stage classifier would probably be a very robust solution.

The solution tracking if not very reliable. I wan to further extend this by incorporating a kalman filter tracking pipeline. I had some previous experience with mean-shift tracking and kalman filters, and I believe some of the ideas there could be exploited to have a very reliable car-tracking and detection.

The picture to picture style of network use here (in this project based on an enconder-like network) can be further extended to also classify lane pixel in another color channel of the output.

The pipeline will fail if you do not provide a typical car-dashboard image, or if the images are terribly bad illuminated.

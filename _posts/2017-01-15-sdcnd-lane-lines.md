---
layout: post
title:  "SDCND Lane Lines Finder"
date:   2017-01-15 16:00:00 +0000
categories: sdcnd opencv
---

In 4th project of [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/drive), the goal is to write a software pipeline to identify the lane boundaries in video from a front-facing camera on a car.
The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

<!--excerpt-->

## Calibration and undistortion

First of all, with OpenCV's function of finding chessboard corners I defined set of points for further camera calibration.
Figure below demonstrates source images of chessboard with drawn corners on it.

![Chessboard images before calibration](/assets/sdcnd-lane-finding/chessboards.png)

By this small snippet of code I prepare `get_undistorted_image` function which is going to be a basis for the pipeline

```
imgsx, imgsy = 5, 4

# Find chessboard corners
for f in glob.glob('camera_cal/calibration*.jpg'):
    # Read and find
    img = mpimg.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny))

    # Store points if corners are found
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)

def get_undistorted_image(img):
    return cv2.undistort(img, mtx, dist, None, mtx)
```

Figure below shows same images but after undistortion procedure.

![Chessboard images after calibration](/assets/sdcnd-lane-finding/chessboards_undist.png)

## Binary tresholding

After long searches of optimal binary masking technique I found next combination:
* from initial RGB image get GRAYSCALE image and HLS images
* apply magniture threshold to GRAYSCALE -- `mag_gray`
* apply magnitude threshold to S channel of HLS image -- `mag_s`
* threshold L channel of HLS image -- take pixels with value >170 -- `l_only`
* threshold H channel of HLS image -- take pixels in range between 12 and 30 (yellow hue range) -- `h_only`
* combine 4 resulting channel in next way: `(mag_gray & h_only) | (mag_s & l_only)`

Lightness channel and Sobel over grayscale work best for white lines while Hue channel and Sobel ober Saturation channel seem to work best for yellow and for most of white lines.

Figure below show masks working separately and combined result.

![Binary tresholding](/assets/sdcnd-lane-finding/binary_masks.png)

## Birds-eye view

Here is my area of perspective transformation that I used for the pipeline:
```
src = np.array([(180, 720), (530, 500), (790, 500), (1230, 720)])
dst = np.array([(180, 720), (180, 0), (1100, 0), (1100, 720)])
```

Figures below show the area of interest on raw image and its resulting birds-eye view.

![Region of interest for bird-eye view](/assets/sdcnd-lane-finding/birdeye_before.png)

![Bird-eye view](/assets/sdcnd-lane-finding/birdeye_after.png)

## Detect line pixels and build a fit polynomial

The alghoritm of line pixels detection works separately for left and right part of the image, trying to find left and right lines respectively.
It has several steps which described below with examples and explanations.
* Split source 1280x720px image into two 640x720px subimages. Consider below working with only one of this image
* Define image width, height, calculate coordinates for slices _(E.g. in 640x720px images I used 10 horizontal slices of 640x72px each)_
* For every slice from bottom to up, do:
    - Detect x-coordinate for window center:
        + If there is detected line in previous frame, take saved start coordinate
        + if overall histogram peak is considered as domination value, suppose it is caused because of well-drawn one vertical lane and take its peak position as start
        + If current slice's histogram peak is high enough (30), start with this peak coordinate
        + If we have history of previous detection, find latest most confident start and use it
        + Else, if nothing of these helped, just use overall peak position
    - When we defined window center, check if its not just a noise by examinating its histogram
    - From left and right of this central position count all slice columns as lane pixels if column's amount of pixels is more than mean histogram value of the slice (to reduce noise)

After pixels detection we still don't sure that there is a line there.
Second part works as follows:
* get all non-zero points (having X and Y coordinates)
* for each unique Y coordinate select related X coordinates and get their mean value
* having set of (Y, X) coordinates, get mean X value and remove points where X deviates from mean more than for 2 standard deviations
* rest points considered to be good
* calculate confidence `self.confidence = len(ygood) / float(self.img_height)` -- more good points we have, more confidence we have. This helps to reduce noise in polynomial building
* if confidence is high enough, build polynomial by this good points, mark frame as detected
* smooth polynomial by next formula `p = p_history.mean() * 0.7 + p * 0.3` _(i.e. keep 30% of current polynomial and 70% of history data)_

Curvature is smoothly calculated as follows:
```
def calculate_curvature(self):
    # If this frame is not confident, return last curvature
    if self.confidence < self.CONFIDENCE_THRESHOLD and self.curvature is not None:
        return self.curvature

    y_eval = self.input_pixels.shape[0]
    xvals = self.line_x_points * self.CURVATURE_XM_PER_PIX
    yvals = self.line_y_points * self.CURVATURE_YM_PER_PIX
    fit = np.polyfit(yvals, xvals, 2)

    # Calculate curvature and save
    curvature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
    self.curvature = self.get_smoothed_value(curvature, self.curvature_history)
    self.curvature_history.append(self.curvature)
    self.curvature_history = self.curvature_history[-self.N_HISTORY_SIZE:]

    return self.curvature
```

Distance calculation is also smoothed:
```
def calculate_distance(self, is_left=True):
    # If this frame is not confident, return last distance
    if self.confidence < self.CONFIDENCE_THRESHOLD and self.distance is not None:
        return self.distance

    mean_lane_points = self.line_x_points.mean()
    if is_left:
        distance = self.img_width - mean_lane_points
    else:
        distance = mean_lane_points
    distance *= self.CURVATURE_XM_PER_PIX

    self.distance = self.get_smoothed_value(distance, self.distance_history)
    self.distance_history.append(self.distance)
    self.distance_history = self.distance_history[-self.N_HISTORY_SIZE:]

    return self.distance
```

## Apply all of this together to build an all-in-one frame for new video!

```
line_left = Line()
line_right = Line()

img = mpimg.imread('test_images/test4.jpg')
plt.figure(figsize=(16, 10))
plt.imshow(process_image(img))
```

![Resulting image](/assets/sdcnd-lane-finding/allin1.png)

## Results

Click the screenshot to see the video on YouTube

### Basic video

<iframe  width="700" height="350" src="https://www.youtube.com/embed/JPNONugHBms" frameborder="0" allowfullscreen></iframe>

### Challenge video

<iframe  width="700" height="350" src="https://www.youtube.com/embed/EUd0ICV4z7w" frameborder="0" allowfullscreen></iframe>

### Harder challenge video

<iframe  width="700" height="350" src="https://www.youtube.com/embed/kDiLxpGVXNw" frameborder="0" allowfullscreen></iframe>

## Reflection

In this project there are 2 main parts which mostly impact the resulting video:
* pixels detection
* anomaly detection in frame flow

Current approach works good for basic and challenge videos but is poor for harder challenge.
As one can see in binary masking output, there should be much better finetuning for darker and lighter areas, as well as finetuning for the region of interest.
Smoothing procedure needs to be reviewed since there are fast curvature changes on harder challenge video.
This harder challenge video also has frames where there is no right line at all, which is also not covered by current alghorithm.

All together, issues raised in harder challenge video demonstrate ways for future improvement of this pipeline

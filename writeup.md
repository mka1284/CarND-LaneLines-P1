# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consists of 5 steps. The steps are the following:

1. Grayscaling: the image is converted to grayscale, so that also colored lines (e.g. the yellow lines) can later be detected

    ![alt text](test_images_output\gray_solidWhiteRight.jpg)

2. Blurring: as a preparation for the Canny Edge Detection, the image is blurred

    ![alt text](test_images_output\blurred_solidWhiteRight.jpg)


3. Canny Edge Detection: based on the blurred image, the edges are detected

    ![alt text](test_images_output\canny_solidWhiteRight.jpg)


4. Cutting: now all pixels laying outside of the region of interest are cut. The region of interest is a triangle ranging from the buttom of the image to almost the center. Cutting is not done before, otherwise it would create ghost edges at the triangle borders.

    ![alt text](test_images_output\cut_solidWhiteRight.jpg)


5. Hough Transform: the hough transform is applied to the image to detect the line segments

    ![alt text](test_images_output\hough_solidWhiteRight.jpg)


6. Draw Lines: the lines which are output of the hough transform are filtered by their slope: for the left lane, a slope of -0.6 and for the right lane 0.6, with a tolerance of 0.2, seemed to be appropriate. Of all the collected lines for the left lanes the average of x, y and the slope is calculated. The line is then drawn from the bottom of the image until the y value of 330

    ![alt text](test_images_output\final_solidWhiteRight.jpg)


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the road is curved too much.
At the moment, the lanes are always assumed to be straight lines, and when the line
is curved a lot, this approximation is not valid.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to approximate the lines by splines: all start and end points and fit splines through it.
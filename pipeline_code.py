# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import os

#os.listdir("test_images/")
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""

    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def calc_line_endpoints(x, y, m):
    """
    Given an equation of a line by x, y, and the slope m, this function calculates the intersections
    of this line with the bottom image border and the upper limit of the lande
    """

    ylow = 540  # height of image
    yhigh = 330  # "upper" limit of lane

    t = y - m * x
    xlow = -(t - ylow) / m
    xhigh = -(t - yhigh) / m

    return (xlow, ylow, xhigh, yhigh)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Given an image (img), and the lines found by the lane detection (lines),
    this function calculates and draws the lane lines on the given image.
    """

    leftlane = []
    rightlane = []

    # average expected slope of the lane lines
    lslope = -0.6
    rslope = 0.6

    # tolerance regarding the slope
    tol = 0.2

    # variables for summing up and averaging the lane segments
    xl = 0;
    yl = 0;
    xr = 0;
    yr = 0;
    ml = 0;
    mr = 0;

    # sort lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            m = (y2 - y1) / (x2 - x1)
            if abs(lslope - m) < tol:
                leftlane.append([line])
                xl = xl + x1 + x2
                yl = yl + y1 + y2
                ml = ml + m
            elif abs(rslope - m) < tol:
                rightlane.append([line])
                xr = xr + x1 + x2
                yr = yr + y1 + y2
                mr = mr + m

    numl = len(leftlane)
    numr = len(rightlane)

    # calculate avg slope m and x, y for left lane
    xlavg = 0.5 * xl / numl
    ylavg = 0.5 * yl / numl
    mlavg = ml / numl

    # calculate avg slope m and x, y for right lane
    xravg = 0.5 * xr / numr
    yravg = 0.5 * yr / numr
    mravg = mr / numr

    xllow, yllow, xlhigh, ylhigh = calc_line_endpoints(xlavg, ylavg, mlavg)
    xrlow, yrlow, xrhigh, yrhigh = calc_line_endpoints(xravg, yravg, mravg)

    cv2.line(img, (int(xllow), int(yllow)), (int(xlhigh), int(ylhigh)), color, thickness)
    cv2.line(img, (int(xrlow), int(yrlow)), (int(xrhigh), int(yrhigh)), color, thickness)


def draw_lines_simple(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function just draws the given lines on the image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Returns an image with hough lines drawn.
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines_simple(line_img, lines)
    return (line_img, lines)


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


######### pipeline functions ##########

def white_mask(img):
    """
    Calculates a white mask, so that only white pixels (with a certain tolerance) remain in the image'
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 15, 255]))

    white_image = cv2.bitwise_and(img, img, mask=mask)

    return white_image


def cut_area(img):
    """
    Makes black everything laying outside of the desired area
    """
    # pts – Array of polygons where each polygon is represented as an array of points.
    vertices = np.array([[(0, 540), (480, 320), (960, 540)]], dtype=np.int32)
    masked_image = region_of_interest(img, vertices)

    return masked_image


def hough_trans(img):
    """
    Applies the hough transform to the image
    """
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_len = 20
    max_line_gap = 20
    hough_image, lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)

    return (hough_image, lines)


def pipeline(initial_image, filename, write_imgs):
    # plt.imshow(initial_img)
    # plt.title('original image')

    # white_masked = initial_img
    # white_masked = white_mask(initial_img)
    # plt.figure()
    # plt.imshow(white_masked)
    # plt.title('white mask')

    gray_image = grayscale(initial_image)
    # plt.figure()
    # plt.imshow(gray_image, cmap='gray')
    # plt.title('grayscale')

    blurred_image = gaussian_blur(gray_image, 5)
    # plt.figure()
    # plt.imshow(blurred_image, cmap='gray')
    # plt.title('gaussian_blur')

    canny_image = canny(blurred_image, 50, 150)
    # plt.figure()
    # plt.imshow(canny_image, cmap='gray')
    # plt.title('canny')

    cut_image = cut_area(canny_image)
    # plt.figure()
    # plt.imshow(cut_image, cmap='gray')
    # plt.title('cut image')

    hough_image, lines = hough_trans(cut_image)
    # plt.figure()
    # plt.imshow(hough_image)
    # plt.title('hough image')

    # plt.figure()
    # hough_img_w_lines = weighted_img(hough_image, img)
    # plt.imshow(hough_img_w_lines)
    # plt.title('img with hough lines')

    # plt.figure()
    # line_img = img.copy() # black image
    line_img = np.zeros((540, 960, 3), dtype="uint8")

    draw_lines(line_img, lines, [255, 0, 0], 8)
    final_image = weighted_img(line_img, initial_image, 0.8, 1., 0.)

    if write_imgs:
        cv2.imwrite('test_images_output/gray_' + filename, gray_image)
        cv2.imwrite('test_images_output/blurred_' + filename, blurred_image)
        cv2.imwrite('test_images_output/canny_' + filename, canny_image)
        cv2.imwrite('test_images_output/cut_' + filename, cut_image)
        cv2.imwrite('test_images_output/hough_' + filename, cv2.cvtColor(hough_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite('test_images_output/final_' + filename, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    return final_image


# iterate through all files in the folder and apply the pipeline functions
for filename in os.listdir("test_images/"):
    image = mpimg.imread('test_images/' + filename)

    final_img = pipeline(image, filename, True)

    plt.figure()
    plt.imshow(final_img)
    plt.title(filename)
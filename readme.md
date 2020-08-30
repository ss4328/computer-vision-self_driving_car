# Computer Vision

## About
Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects — and then react to what they “see.” 

In order to dig deeper in Computer vision, we must understand the association with machine learning. Computer vision, is more than machine learning applied. It involves tasks as 3D scene modeling, multi-view camera geometry, structure-from-motion, stereo correspondence, point cloud processing, motion estimation and more, where machine learning is not a key element.

Simply said, using some algorithms, we could program a program to 'see' (and therefore act) on the basis of provided data, which could be images, or videos(interpreted mostly as a set of images in quick succession).

Following a top down approach, I'll learn Computer vision by attempting to make a self-driving car. The undertaking is bold, as I know next to nothing about computer vision, yet exciting as this is an amazing problem, and has so much scope to teach!

## Data
1. The image of a road - in data/lane_image


## Process
1. lane_image will be used for basing the perception of the car. We'll try to implement a lane-finding algorithm. 
2. We'll try canny edge detection technique: identifyihng sharp changes in intensity in adjacent pixels. It's a multistep process:
- Gradient change: measure of change in brightness over adjacent pixels. This helps us identify edges in image.

### Canny edge detection
1. Convert to grayscale - Images are made of 3-channel colors (RGB), we convert to grayscale for 1-channel intensity only. By using grayscale, processing a single channel is faster.
```python

gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
```
2. Gaussian blur: Reduce noise and smoothening. We need to do this to reduce false-positive edges. This is done by replacing a pixel value with the weighted average of it's neighboring pixel values. (This smoothenes the image mostly but if more info is needed, look up kernel convolution) 
```python
 blur = cv2.GaussianBlur(gray, (5,5), 0)
 ```
 (step is optional if we use the canny function as it automatically uses it inside its implementation)
3. Canny method: Performs derivative in both x, y directions. This gives a 'change in intensity', which is used to find out canny edges. Then traces the strongest gradient changes in white color. Documentation read recommended. 

```python
canny = cv2.Canny(blur, 50, 150) #args = (image, low_threshold, High_threshold)
```

## Region of Interest
the region on interest is hte lane we're driving in on right now. 

First things first, let's change the image showing from cv2 to matplotlib. This will allow us to choose the region of interest based on direct pixel values from image. This will come off as a triangle. 

```python
# return the enclosed regions of our FOV (triangular in space)
def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
	[(200, height), (1100, height), (550, 250)]
	]) #triangle by vertices
	mask = np.zeros_like(image) #same shape as image

	#fill this mask with our polygon in white color
	cv2.fillPoly(mask, polygons, 255)
	return mask
```

## Getting driveable area
Our AOI, or area of interest is filled with 255 (or bitwise 111111) and the other part of mask has 0 value( bitwise OOOOOOO). To get the lanes inside AOI, we'll simply do a Bitwise OR between the AOI and original image. This will just get us lanes in AOI and anything else will get blacked out! Pretty neat, huh?
```python
# return the enclosed regions of our FOV (triangular in space)
def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
	[(200, height), (1100, height), (550, 250)]
	]) #triangle by vertices
	mask = np.zeros_like(image) #same shape as image

	#fill this mask with our polygon in white color
	cv2.fillPoly(mask, polygons, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image
```

## Hough Transform
So far, we've identified edges of interest and isolated AOI. Now, we need to get the straight lines in the image, and so get lanes. For this we'll do a simple Hough Transform. 

Whenever we see a collection of points, and told that these points are connected by some lines, ask the question "what's the line?" There are many lines that can pass a point individually, but only few lines with same slope/intercepts. We can determine that by looking at the point of interception in hough space. That point of interception in hough space will give us line that passes both points! 

How is this relevant? This idea of identifying lines from a series of points will give us lines (lanes in our case). Best fitting lines wins. 

```python
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #2 pixels, 1 radian, 100 intersections as threshold

```

## Applying lines
Now we have found out the lines that fit the lanes in our image. Next, we'll need to polish those multiple lines to left/right lane lines, and apply on the given image. 


```python

def display_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines: 
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)

	return line_image

#function for cropping the highlighted lines to 3/5 total height
def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1-intercept)/slope)
	x2 = int((y2-intercept)/slope)
	return np.array([x1,y1,x2,y2])


def average_slope_intercept(image, lines):
	left_fit=[]
	right_fit=[]
	for line in lines: 
		x1, y1, x2, y2 = line.reshape(4)
		#fit linear polynomial to the points above
		parameters = np.polyfit((x1, x2), (y1,y2), 1)
		slope = parameters[0]
		intercept = parameters[1]
		if(slope<0):
			left_fit.append((slope,intercept))
		else:
			right_fit.append((slope,intercept))
	print(left_fit)
	print(right_fit)
	left_fit_avg = np.average(left_fit, axis=0)
	right_fit_avg = np.average(right_fit, axis=0)
	print(left_fit_avg, 'left')
	print(right_fit_avg, 'right')
	left_line = make_coordinates(image, left_fit_avg)
	right_line = make_coordinates(image, right_fit_avg)
	return np.array((left_line, right_line))

#and finally
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #2 pixels, 1 radian, 100 intersections as threshold
averaged_lines = average_slope_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
final_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)
cv2.imshow('result', final_image)
cv2.waitKey(0)

```


## Running on a Video: 
Next, we can just get a video in openCV, then run our algorithm in loop. 

```python
cap = cv2.VideoCapture('testVideo.mp4')
while(cap.isOpened()):
	_, frame = cap.read()
	#returns image as a m,iltidimentional numpy array, containing relative intensity of each pixel
	lane_image = np.copy(frame)
	canny_image = canny(lane_image)
	cropped_image = region_of_interest(canny_image)

	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #2 pixels, 1 radian, 100 intersections as threshold
	averaged_lines = average_slope_intercept(lane_image, lines)
	line_image = display_lines(lane_image, averaged_lines)
	final_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)
	cv2.imshow('result', final_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
```

## Conclusion
That definitely took more work than I expected. Here's the gist:
1. We get a video-frame/image
2. Grayscale that frame
3. Apply gaussian filter for blurring
4. Then canny edge detection algorithm
5. Then we use the Hough transformations to get lines in the video
6. Finally, we average those lines (easier said than done)
7. Display the lane lines over given frame

I'll continue to use this in a NN to make a ()

## Contact
- E-mail: shiv.suhane@gmail.com
- [LinkedIn](https://www.linkedin.com/in/shivansh-suhane/)

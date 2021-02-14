# speed_estimation
There are two videos of a fixed traffic camera facing to both sides of the road. Some physical measurements are also provided in the sample image. The problem is to calculate the speed of each vehicle (mostly cars) in the scene.

# **Calculate the Speed of each Vehicle**



Before providing the solution, I need to mention that I believe a real AI person has to know all of his/her assets and tools (in my case computer vision, machine learning and AI techniques). Then, he/she must understand the problem with critical thinking to break down the issue into smaller parts and answer every subpart. In every case of the domain challenge, domain knowledge can help us to find a better solution.  Finally, I believe that the first solution, in most cases, is not the perfect one; it just makes the problem clearer. An AI person can detect the solution's weakness and gradually respond to it with proper improvement in a couple of stages. Due to this fact, I think that the current answer is not the ultimate one. It could be improved in various ways (selecting better object detector, better distance estimator, better speed calculation equation.)



![Diagram  Description automatically generated](https://lh5.googleusercontent.com/v_RAsjCdjCo7ioIDaU8jzYW723QlqSvBUsK4ApTAhRzr3g7_wdvg9PQeQI80ycoRJiK--AZdIwCOsngjjcpcvbfRbbk5OvGDHkVhWv1_f8D49lWMuai1COIe5KIhRIgs2dw0zU0)

Fig.1 The overall solution for the speed calculation problem.



Fig.1 illustrates the overall solution for calculating the vehicle's speed by having a fixed traffic camera. To solve this issue, at first, I have to breakdown the task into some subtasks. There are three main tasks in this project as it is shown in the Fig1.,

- Vehicle Detection

We utilized YOLOv4[1] to identify vehicles in each frame.

- Vehicle Tracking - ( assigning IDs to vehicles )

We have used DeepSORT[2] to distinguish and track every single vehicle.

- Speed Calculation

We calculate the distance moved by the tracked vehicle in a second, in terms of pixels, so we need pixel per meter to calculate the distance travelled in meters.

As shown in Fig.1, I utilize the latest well-established YOLOv4 model (recently, YOLOv5[3] is proposed as well) for detecting the car in every frame. I selected YOLOV4 due to its speed and accuracy performance. Then, I applied the DeepSORT, which was introduced in 2018 to assign every single vehicle in the frame. To my understanding, the accuracy of DeepSORT is reasonably good enough for this application. Finally, we must use the tracking information to calculate the speed for every vehicle in the frame. This is the most challenging part of the project. To detect the speed of a vehicle, we have two potential solutions:

1. Measuring the speed by converting the distance moved from the pixel to the meter. And then use the following equation (v = x/t), which is the movement divided by the time needed for that.
2. The second approach could be training a deep learning model to estimate the distance in every frame, the subtracting the distance from each other and finally use the above speed equation to calculate the speed.

With my understanding and knowledge, the second one could be a more robust one. However, it has some processing speed issue, which can be handled by mixing the distance measurement with the YOLOv4 or any deep learning model for vehicle detection. This model is more robust due to the model can learn to estimate the perspective distortion within the frame and calculate the distance. A KITTI dataset [4] can be used to train a model and then fine-tune it with our special application. However, for this project, due to the limited time, I just used the former solution. The first solution has two big issues:

1. It would be so sensitive to the exact pixel location; a couple of pixels cause a fluctuation in the speed.

To address this problem, I defined a length for every vehicle (Car =3.5 m, Truck= 4.5m, and Bus =6m), and also instead of calculating the speed per frame, I calculate the movement in every five frames, and then calculate the speed. This would cause the model to become more robust to the bounding box fluctuation from the YOLOv4.



1. The most crucial issue in the first approach is Perspective distortion which is caused by a fixed camera. We can solve this issue a bit by helping OpenCV functions (getPerspectiveTransform, warpPerspective), but again I should mention that this is not the ultimate solution.



For fixing the perspective distortion, I defined two RoI areas, one for the left-hand side of the camera, and the other for every frame's right-hand side. To measure the speed of a car, we must know the exact location of a car in every frame, for that, I just consider the centre of each vehicle bounding box the car location in every frame. It will also cause a lot of problems due to the bounding box size changing. If we want to consider this solution as a final solution, we need to spend time on this matter to solve the bounding box's vibration from YOLOv4. The other thing that I defined was a list of 1000 cars to capture this information:

- The **vehicle id**, **previous position**, **next position** after five frames, **counter for the number of times** the vehicle was in the video, and **speed**.

With having all this information, we can calculate the speed with the following operation. d_pixels gives the pixel distance travelled by the vehicle in five frames of our video processing. To estimate speed in any standard unit first, we need to convert d_pixels to d_metres.

Now, we can calculate the speed(speed = d_meters * fps * 3.6). d_meters is the distance travelled in one frame. We have already calculated the average fps during video processing. So, to get the speed in m/s, just (d_metres * fps) will do. We have multiplied that estimated speed by 3.6 to convert it into km/hr.



For the implementation of this project, we need these main tools:

- Python
- Tensorflow
- OpenCV
- YOLOv4 pre-trained model









**Instruction:**

1. Download the file from this google cloud link:



1. cd (change directory) into vehicle-speed-check cd vehicle-speed-check
2. Create virtual environment python -m venv 
3. Activate virtual environment ./venv/bin/activate
4. Install requirements pip install -r requirements.txt
5. Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT



1. Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.



1. To implement the object tracking using YOLOv4, first, we convert the weights into the corresponding TensorFlow model, which will be saved to a checkpoints folder. We need to run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.



\# Convert darknet weights to TensorFlow model

python save_model.py --model yolov4



1. Run object_tracker.py in spyder or terminal:

\# Run yolov4 deep sort object tracker on video

python object_tracker.py





Reference link:

1. https://arxiv.org/abs/2004.10934
2. https://arxiv.org/abs/1703.07402
3. https://github.com/ultralytics/yolov5
4. http://www.cvlibs.net/datasets/kitti/

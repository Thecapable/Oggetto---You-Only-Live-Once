# Oggetto-You Only Live Once

This project intends to identify objects in a frame, be it a real-time video or an image. After detection, the results are shown in the form of bounding boxes along with the name of the object and also audio output which also speaks out names of all the objects in the image. We'll use YOLO pre-trained model for detection of objects as it has been already trained on a large dataset containing various classes of objects, and hence, makes our job less time-taking. 

![Methodology and Algorithm](https://github.com/Thecapable/Oggetto---You-Only-Live-Once/blob/master/Methodology.png)

You may check out this link for knowing more about YOLO and how to implement it- https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection

# Requirements
- Python 2.7 or above
- OpenCV
- Tensorflow

# Step-1 (Download the Darkflow Repository
- https://github.com/thtrieu/darkflow
- extract the files on your local device

# Step-2 (Building the Library)
- Open the command prompt and type the following command

 * "python setup.py build_ext --inplace" 
 
 # Step-3 (Downloading a weights file)
 - You need to download the pre-trained weight file- https://pjreddie.com/media/files/yolov2.weights
 This particular weights file is the most common and hence I recommend you to use this. You may try any other weights of your liking from- https://pjreddie.com/darknet/yolov2/
 - After downloading the weights, you have to put the file in a folder named 'bin' that you have to create inside the folder where you extracted the Darkflow repo (Step-1).
 
 # Step-4 (Creating Interface)
 - To keep things simple, I've used the Python library called 'Tkinter'. You need to run the python file named 'OGGETTO' to access the following interface:
 
 ![Snapshot of Interface](https://github.com/Thecapable/Oggetto---You-Only-Live-Once/blob/master/interface%20snippet.PNG)
 
 -You need to install an additional library -'pyttsx3'to synthesize text into audio you can hear. It works offline.
 Use pip to install the package. If you are in Windows, you will need an additional package, pypiwin32 which it will need to access the native Windows speech API.

```
#in cmd
 pip install pyttsx3
 pip install pypiwin32  # Windows only
```


# Step-5(Running the code)
- Run [__'OGGETTO.py'__](https://github.com/Thecapable/Oggetto---You-Only-Live-Once/blob/master/OGGETTO.py) file
- Upon clicking __Click here to select an image from the PC__ option, the [__ImgInput.py__](https://github.com/Thecapable/Oggetto---You-Only-Live-Once/blob/master/ImgInput.py) file runs and gives you the option to select an image from your pc to detect objects in it and thanks to __Ppyttsx3__, the output is spoken by your pc saying something like- _'The Objects are: dog, cat, bottle'_ along with giving out an image with bounding boxes around detected object in the original image with their names.
- Upon clicking __Click here for real-time detection__ option, the [__Try1.py__](https://github.com/Thecapable/Oggetto---You-Only-Live-Once/blob/master/Try1.py) file runs and objects are detected in real-time using your webcam by default.

# Product Scope
*	IMAGE RETRIEVAL, SECURITY AND VIDEO SURVEILLANCE 
*	CCTV MONITORING WITHOUT THE NEED OF HUMAN BEINGS
*	APP FOR LOW-VISION COMMUNITY DUE TO HANDS-FREE APPROACH
*	MANUFACTURING INDUSTRY( FOR EXAMPLE- DETECTING ONLY CIRCULAR OBJECTS, OR MACHINE INSPECTION, ETC)


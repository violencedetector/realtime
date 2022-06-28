# realtime
## Welcome to the first open source project for real time violence detection.

Thanks for your interesting in the project, in case of questions please mail us: contact@violencedetector.org 

### Project briefing
This is a real time violence detector project which  will leverage the development of a family of new products for smart security devices.
Today this technology is in the early stages of industrialization, therefore only big organizations have resources to develop it, that's why I've decided to opensource it. We can change it together, since this is too relevant to be restricted.
The framework is written in python with openCV, the intention is keeping it as much multi plattform as possible.


### What the framework is?
**It is:**
- A real time violence detector framework built using computer vision, it comprehends but is not limited to:
- Dataset and it's composition,
- Image preprocessing,
- Network architecture: object detection / image classification,
- Keypoint extraction,
- Decision rules for the detection,
- **Software must be hardware independent as much as much as possible**

**It is not:**
- Final product development,
- Hardware related project,
- Privacy related discussion,
- It did not supposed to cover:
- Software not related to the detection,
- OS related matters as drivers, packages or external dependencies

### Our target
Develop an opensource framework using computer vision to detect violence in real time.
This real time violence detector framework will leverage the development of a family of new products for smart security devices.

### Applications
General surveillance, anti terrorism, crowd management, assault detection, goods transportation protection...

Potential connection to police network for automatic crime detection and prevention.

### Foundations
The framework will take care about processing the image frame(s) and detect the existence of violence. With a simple output yes or no.

It is composed by python detection class file and a frozen model file, the target is being as much as possible platform invariant.

Framework intend to run in 2 modes: 
- A) Framework runing at the same image acquisition device 
- B) In a remote server via API call, sending images and returnind detection result

To keep it simple at the first moment the focus will be in the A mode, since the B mode should(at least this is the expectation) to re-utilize the majority of the code.

### Types of Violence targeted:
- Assault,
- Fight,
- Lynching,
- Hand gun,
- Knife,
- Terrorism,
- General violent behavior

### What is next?
If you want to contribute with the project, please take a look in the [CONTRIBUTORS.md](CONTRIBUTORS.md) where you can find relevant additional information about the project.

In case of questions, please contact us contact@violencedetector.org

--------------------------------------------------------------------------------------------------------------------------------------------------
## Step by step for the article demo:

To run the method in the article(https://www.igi-global.com/article/simple-approach-for-violence-detection-in-real-time-videos-using-pose-estimation-with-azimuthal-displacement-and-centroid-distance-as-features/304462) you firstly need to install the OpenPose framework.

There are multiple installation options, this is the official Repo:
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md

In my case, I've used one other repo with a tensorflow implementation(https://github.com/ZheC/tf-pose-estimation) which I've cloned:
```
 git clone https://www.github.com/ildoonet/tf-openpose
 cd tf-openpose
 pip3 install -r requirements.txt
```
I've created one additional method under the main class TfPoseEstimator for supporting the feature extrataction. 
You can replace the standard file by the enhanced one(estimator.py) available here in this repo, simply overwriting it under its location(Pose_Estimation/tf-pose-estimation/tf_pose).

Before running the test(article.py) you need 3 steps:
- 1-Copy the frames(*) to be processed, I've added a test sample of violence and non-violence under the image folder. 
- 2-Copy the trained model(best_acc_final.keras) to the tf-pose-estimation folder.
- 3-Copy the python script article.py
Now you can run the test passing the frames location as a parameter like that:
```
 python3 article.py --file cam22.mp4*
```
Note: For keeping this example simple, the code is limited to 2 individuals, nevertheless there is no limitation for the method.

The last frame will be presented with the OpenPose points, after clicking ESC you will see the array of the inference result for each individual in the timeseries submitted.

(*) Images extracted from the dataset(which was not the same used for the training): 
M. Bianculli, N. Falcionelli, P. Sernani, S. Tomassini, P. Contardo, M. Lombardi, A.F. Dragoni, A dataset for automatic violence detection in videos, Data in Brief 33 (2020). doi:10.1016/j.dib.2020.106587.


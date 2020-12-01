# realtime
Welcome to the first open source project for real time violence detection.

We are a group of people who believes that technology can make the world safer.
This real time violence detector framework will leverage the development of a family of new products for smart security devices.
If you want to contribute for making the world a safer place join us!
Today this technology is in the early stages of industrialization, therefore only big organizations have resources to develop it, that's why I've decided to opensource it. We can change it together, since this is too relevant to be restricted.

Technology democratization is the answer!

What is the framework?
It is:
A real time violence detector framework built using computer vision, it comprehends but is not limited to:
Dataset and it's composition,
Image preprocessing,
Network architecture: object detection / image classification,
Keypoint extraction,
Decision rules for the detection,
*Software must be hardware independent as much as much as possible
It is not:
Final product development,
Hardware related project,
Privacy related discussion,
It did not supposed to cover:
Software not related to the detection,
OS related matters as drivers, packages or external dependencies

Our target
Develop a opensource framework using computer vision to be used for real time violence detection.
This real time violence detector framework will leverage the development of a family of new products for smart security devices.

Applications
General surveillance, anti terrorism, crowd management, assault detection, goods transportation protection...

Potential connection to police network for automatic crime detection and prevention.

Foundations
The framework will take care about processing the image frame(s) and detect the existence of violence. Simple output yes or no.

It is composed by python detection class file and a frozen model file.

The target is being as much as possible platform invariant.

Types of Violence targeted:
Assault,
Fight,
Lynching,
Hand gun,
Knife,
Terrorism,
General violent behavior

Violence detection class:
Image(s) preprocessing and normalization,
Region of interesting and foreground extraction,
Load model,
Prediction based on model,
Prediction result interpretation and result,
Return yes/no


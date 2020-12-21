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

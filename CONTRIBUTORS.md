## 👍🎉 At first thanks for taking the time to contribute! 🎉👍

### Below you can find the guidelines. Use your best judgment, and feel free to propose changes to this document in a pull request. 

### What should I know before I get started?

This is a fresh new project, so who join now will be able to shape the framework thru contributions that are not limited to the code, at this moment the conceptual discussion is veey important.

Please check the documentation below, in case of any question please send a mail to contact@violencedetector.org

### How Can I Contribute?

#### Being part of the discussions
- As we are starting, this is definitely the best way to contribute, please take a look at the issues with the [DISCUSSION] mark.

#### Suggesting Enhancements
- Please use the standard pull request process, please feel free to suggest any other project improvement.

#### Working in the issues
- Please use the standard issue process.

## Documentation
### class violence_detector

#### Methods:

##### Init
Load model and sets internal variables

Input: N/A

Output: N/A
##### Preprocess
This method is responsible for resizing the image according to the model requirements then apply normalization or any other preprocessing method.

Input: Raw image(s)

Output: Preprocessed image(s)
##### Predict
Perform violence detection on the image(s).

Input: Preprocessed image(s)

Output: Top 3 prediction tuple(class ID, Description and probability)
##### Decision
Decides upon the criteria if violence is present or not based on the prediction results.

Input: Prediction tuple

Output: True for violence detected and False for not



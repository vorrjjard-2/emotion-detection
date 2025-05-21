# CNN For Facial Emotion Detection with PyTorch
Project Status : **Archived**

A convolutional neural network (CNN) built with `pytorch` to classify emotional states in still images. Also connects with webcams using `opencv` to classify emotions in real time.

Insert DEMO : 

___
### Project Structure
```text
- data/                 # Data Preprocessing
- models/               # Model Architecture
- utils/                # Helper Functions
- train.ipynb           # *Main Training Script with camera*
- requirements.txt
- README.md
```
___
### Dataset
The FER2013 dataset was selected, and split into train, validation, and test splits. There are a total of around 34,000 images in the dataset, with seven labels corresponding to seven emotions.

The dataset was taken from [Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013).
___
### Model Architecture
The CNN was built with three(3) convolutional blocks followed by two fully connected layers with softmax resulting in a seven item tensor with a probability distribution for the emotion currently being predicted. 

**Cross entropy loss** was selected as the loss metric of choice.
**Adam** was selected as the optimizer.

**Convolutional Block/s**
```python3
self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )
```
```python3
self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
```
```python3
self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
```
**Fully Connected Layers**
```python3
self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )
```
___
### Results
An accuracy of around **70%** was observed after completing training on the test dataset of the FER2013 set.
___
### How to Run
```python3
# 1. Clone the repository
git clone https://github.com/vorrjjard-2/emotion-detection.git

# 2. Install the required dependencies
pip install -r requirements.txt

# 3. Run the train script on Jupyter. The last code cell activates the camera integration.
jupyter lab
```
___
### Possible Improvements
The project is still quite basic and could use improvements to the architecture to enhance learning. Some improvements I could see happening are : 
1. Hyperparameter Tuning
2. Implementing Transfer Learning
3. Implementing Temporal Smoothing with openCV
___
You can read my ML blog here :
https://medium.com/@vorrjjard_

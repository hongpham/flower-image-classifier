# IMAGE CLASSIFIER FOR FLOWER 
Train and predict flower name based on the image using pretrained ML model

## Train new model 
Train.py creates and trains a new model based on popular torch.models densenet121 or vgg16. It then save a checkpoint of this new model so that you can use this checkpoint for prediction or future model training. You can run this program with GPU or CPU option (FYI: choosing CPU option will take the program much longer to complete)

## Predict flower name and probability
Predict.py load the checkpoint created by Train.py and perform prediction based on an image of the flower. It then print out flower name, and the probability (or confident level) that the image is infact the flower that has the predicted name

### Sample Jupyter Notebook
A sample Julyter Notebook 'Image Classifier.ipynb' is also provided.

# Crack_detection_part_metal_python2
Python 2 implementation for android app development
  A python flask app running on local host.
  Due to computation limitation, images if trimmed to 512x512 pixels.

## Pre-requisite nstall requirements
  `pip install -r requirements.txt`

## Run the program
  ### Some preprocessing to image
  #### Structure of folders :
  * Download the dataset from [DataSet](https://drive.google.com/open?id=168I7Gg0AMEZ_ne6mU3wx4puUxMs4TFse) and extract the contents to the folder "YE358311_Fender_apron"
  * Dataset Directory containing "normal" and "defect" folders
    * dataset ---> "/YE358311_defects/YE358311_Crack_and_Wrinkle_defect/"
    * dataset ---> "/YE358311_Healthy/"
  * Destination Directory containing "train" and "test" folders
    * data ----> train ----> {"normal", "defect"} subfolders
    * data ----> test ----> {"normal", "defect"} subfolders

  ### Training a simple CNN classifier (3 Conv + 1 FC)
  * `python2 main_train.py` --- Start the training application server(local)
  * `http://127.0.0.1:5000/crack_detection_train` --- Train the model
  * `python2 main_predict.py` --- Start the testing application server(local)
  * `http://127.0.0.1:5000/crack_detection_test` --- Opens an hmtl to upload the image and predict if Defective or Healthy

## Accuracy Metrics
  Validation accuracy and loss
  * Training loss : 0.1695
  * **Training accuracy** : 97%
  * Validation/Test loss : 0.2475
  * **Validation/Test accuracy** : 92.4%

## Need for Improvements
* Next commit is to submit an andriod one page app to interact with the python flask API, showing upload, train and test functionality just like the webapp above.
* Image Preprocssing/Data Preparation
  * Since cracks are of less area as compared to image and noise, will introduce dropouts to improve accuracy
  * Preparing a object extraction module (For removing the rest of noise to improve accuracy)
  * Preparing a background color update module (For easy extraction of metal part after grayscale conversion)
* Trainng Improvements
  * Using a pretrained model such as VGG16 trained on ImageNet
  * Using model ensembles such as CNN+SVM (rbf kernel), Gauss filter+LBP+SVM(rbf kernel) etc that have proved improving accuracy

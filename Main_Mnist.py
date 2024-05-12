import Config
#from NeuroUtils import Core
#from NeuroUtils import ML_assets as ml

#For working with unreleased version directly from local repository, to test things out
import sys
sys.path.insert(0, "C:\\Users\\Stacja Robocza\\Desktop\\NeuroUtils\\NeuroUtils")
import Core
import numpy as np
import os
import pandas as pd

#1
#Creating Class of the project, putting parameters from Config file
Mnist = Core.Project.Classification_Project(Config)

#2
#Initializating data from main database folder to project folder. 
#Parameters of this data like resolution and crop ratio are set in Config
Mnist.Initialize_data()
####################################################



####################################################
#3
#Loading and merging data to trainable dataset.
Mnist.X_TRAIN = np.load(os.path.join(Mnist.DATA_DIRECTORY , "x_train.npy"))
Mnist.X_TEST = np.load(os.path.join(Mnist.DATA_DIRECTORY , "x_test.npy"))
Mnist.Y_TRAIN = np.load(os.path.join(Mnist.DATA_DIRECTORY , "y_train.npy"))
Mnist.Y_TEST = np.load(os.path.join(Mnist.DATA_DIRECTORY , "y_test.npy"),allow_pickle=True)

Mnist.N_CLASSES = len(Mnist.Y_TRAIN[0])
Mnist.DICTIONARY = [(element , str(element))for element in np.arange(Mnist.N_CLASSES)]

#4
#Processing data by splitting it to train,val and test set and data augmentation.
#Parameters of augmentation and reduction are set in the Config file
Mnist.Process_data()

#5
#Initialization of model architecture from library. 
#Model architecture is specified in the config
#This step can be skipper and you can provide your own compiled model by:

Mnist.Initialize_model_from_library()

#6
#Training of the model. It can load previously saved data from project folder
#or train from scratch. Parameters of training are set in Config file
Mnist.Initialize_weights_and_training()

#7
#Showing results of the training and evaluates the model
#Parameters of results can be set in Config file
Mnist.Initialize_resulits()

#8
#Making submission
sample_submission = Mnist.Generate_sample_submission()
sample_submission.to_csv(os.path.join(Mnist.MODEL_DIRECTORY,"sample_submission.csv"), index = False)

















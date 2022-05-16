#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  train-model.py
#  
#  Copyright 2022 Stephen Stengel <stephen.stengel@cwu.edu>
#  

print("Loading imports...")

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time
import cv2

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, classification_report
from models import createHarlowModel, simpleModel, inceptionV3Model
from keras import callbacks

print("Done!")

LOADER_DIRECTORY = os.path.normpath("../animal-crossing-loader/")
TRAIN_DIRECTORY = os.path.join(LOADER_DIRECTORY, "dataset", "train")
VAL_DIRECTORY = os.path.join(LOADER_DIRECTORY, "dataset", "val")
TEST_DIRECTORY = os.path.join(LOADER_DIRECTORY, "dataset", "test")

CLASS_BOBCAT = 0
CLASS_COYOTE = 1
CLASS_DEER = 2
CLASS_ELK = 3
CLASS_HUMAN = 4
CLASS_NOT_INTERESTING = 5
CLASS_RACCOON = 6
CLASS_WEASEL = 7

CLASS_BOBCAT_STRING = "bobcat"
CLASS_COYOTE_STRING = "coyote"
CLASS_DEER_STRING = "deer"
CLASS_ELK_STRING = "elk"
CLASS_HUMAN_STRING = "human"
CLASS_RACCOON_STRING = "raccoon"
CLASS_WEASEL_STRING = "weasel"
CLASS_NOT_INTERESTING_STRING = "not"

CLASS_NAMES_LIST_INT = [CLASS_BOBCAT, CLASS_COYOTE, CLASS_DEER, CLASS_ELK, CLASS_HUMAN, CLASS_NOT_INTERESTING, CLASS_RACCOON, CLASS_WEASEL]
CLASS_NAMES_LIST_STR = [CLASS_BOBCAT_STRING, CLASS_COYOTE_STRING, CLASS_DEER_STRING, CLASS_ELK_STRING, CLASS_HUMAN_STRING, CLASS_NOT_INTERESTING_STRING, CLASS_RACCOON_STRING, CLASS_WEASEL_STRING]

TEST_PRINTING = True

IMG_WIDTH = 100
IMG_HEIGHT = 100
# ~ IMG_WIDTH = 200
# ~ IMG_HEIGHT = 150
# ~ IMG_WIDTH = 400
# ~ IMG_HEIGHT = 300
# ~ IMG_WIDTH = 300
# ~ IMG_HEIGHT = 225
IMG_CHANNELS = 3

IMG_SHAPE_TUPPLE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE = 32	
EPOCHS = 20


def main(args):
	listOfFoldersToDELETE = []
	deleteDirectories(listOfFoldersToDELETE)
	
	#base folder for this run
	ts = time.localtime()
	timeStr = "./%d-%d-%d-%d-%d-%d/" % (ts.tm_year, ts.tm_mon, ts.tm_mday, ts.tm_hour, ts.tm_min, ts.tm_sec)
	timeStr = os.path.normpath(timeStr)
	
	# Folders to save model tests
	simpleFolder = os.path.join(timeStr, "simple")
	harlowFolder = os.path.join(timeStr, "harlow")
	inceptionFolder = os.path.join(timeStr, "incpetionV3")
	modelBaseFolders = [simpleFolder, harlowFolder, inceptionFolder] #Same order as the modelList below!
	makeDirectories(modelBaseFolders)
	
	# train_ds is for training the model.
	# val_ds is for validation during training.
	# test_ds is a dataset of unmodified images for testing the model after training.
	train_ds, val_ds, test_ds = getDatasets(TRAIN_DIRECTORY, VAL_DIRECTORY, TEST_DIRECTORY)
	
	if TEST_PRINTING:
		printSample(test_ds)
	
	shape = IMG_SHAPE_TUPPLE
	batchSize = BATCH_SIZE
	numEpochs = EPOCHS
	modelList = [simpleModel(shape)] # ~ [simpleModel(shape), createHarlowModel(shape), inceptionV3Model(shape)]

	# This for loop can be compartmentalized into helper functions.
	# There will be one wrapper function to perform k-folds
	# something like performExperiment -> performKfolds -> contents of this for loop.
	for i in range(len(modelList)):
		thisModel = modelList[i]
		thisModel.summary()
		thisOutputFolder = modelBaseFolders[i]
		print("Training model: " + thisOutputFolder)
		thisCheckpointFolder = os.path.join(thisOutputFolder, "checkpoint")
		thisMissclassifiedFolder = os.path.join(thisOutputFolder, "misclassifed-images")
		foldersForThisModel = [thisOutputFolder, thisCheckpointFolder, thisMissclassifiedFolder]
		makeDirectories(foldersForThisModel)
		
		#save copy of source code that created the output
		os.system("cp train-model.py   " + os.path.join(thisOutputFolder, "train-model.py"))
		
		myHistory = trainModel(thisModel, train_ds, val_ds, thisCheckpointFolder, numEpochs)
		print("Creating graphs of training history...")
		strAcc, strLoss = saveGraphs(thisModel, myHistory, test_ds, thisOutputFolder)
  
  		#workin on this.
		stringToPrint = "Epochs: " + str(numEpochs) + "\n"
		stringToPrint += "Image Shape: " + str(IMG_SHAPE_TUPPLE) + "\n\n"
		stringToPrint += evaluateLabels(test_ds, thisModel, thisOutputFolder, thisMissclassifiedFolder, batchSize)
		stringToPrint += "Accuracy and loss according to tensorflow model.evaluate():\n"
		stringToPrint += strAcc + "\n"
		stringToPrint += strLoss + "\n"
		
		statFileName = os.path.join(thisOutputFolder, "stats.txt")
		printStringToFile(statFileName, stringToPrint, "w")
		print(stringToPrint)

	return 0


# model.predict() makes an array of probabilities that a certian class is correct.
# By saving the scores from the test_ds, we can see which images
# cause false-positives, false-negatives, true-positives, and true-negatives
def evaluateLabels(test_ds, model, outputFolder, missclassifiedFolder, batchSize):
	print("Getting predictions of test data...")
	testScores = model.predict(test_ds, verbose = True)
	actual_test_labels = extractLabels(test_ds)
	
	#Get the list of class predictions from the probability scores.
	p_test_labels = getPredictedLabels(testScores)
	
	saveMisclassified(test_ds, actual_test_labels, p_test_labels, missclassifiedFolder, batchSize)
	
	printLabelStuffToFile(testScores, actual_test_labels, p_test_labels, outputFolder) # debug function
	
	outString = "Confusion Matrix:\n"
	outString += "Bobcat(0), Coyote(1), Deer(2), Elk(3), Human(4), Not Interesting(5), Raccoon(6), Weasel(7)\n"
	
	cf = str(confusion_matrix(actual_test_labels, p_test_labels))
	cf_report = classification_report(actual_test_labels, p_test_labels, digits=4)
 
	outString += cf + "\n" + cf_report + "\n"	
	
	#Make a pretty chart of these images?
	
	return outString


# Saves all missclassified images
def saveMisclassified(dataset, labels, predicted, missClassifiedFolder, batchSize):
	cnt = 0
	for img, _ in dataset.take(-1):
		for i in range(batchSize):
			if labels[cnt] != predicted[cnt]:
				myImg = np.asarray(img)
				thisActualName = CLASS_NAMES_LIST_STR[labels[cnt]]
				thisPredictedName = CLASS_NAMES_LIST_STR[predicted[cnt]]
				thisFileString = 
						"actual_" + thisActualName
						+ "_predicted_" +  thisPredictedName
						+ "_" + str(cnt) + ".jpg"
				path = os.path.join(missClassifiedFolder, thisFileString)
				saveThis = np.asarray(myImg[i]) * 255
				cv2.imwrite(path, saveThis)
    
			if cnt < len(labels) - 1:		
				cnt += 1
			else:
				return
    

# Creates the necessary directories.
def makeDirectories(listOfFoldersToCreate):
	for folder in listOfFoldersToCreate:
		if not os.path.isdir(folder):
			os.makedirs(folder)


def deleteDirectories(listDirsToDelete):
	for folder in listDirsToDelete:
		if os.path.isdir(folder):
			shutil.rmtree(folder, ignore_errors = True)	


# add checkpointer, earlystopper?
def trainModel(model, train_ds, val_ds, checkpointFolder, numEpochs):
	checkpointer = callbacks.ModelCheckpoint(
		filepath = checkpointFolder,
		monitor = "accuracy",
		save_best_only = True,
		mode = "max")
	
	earlyStopper = callbacks.EarlyStopping(monitor="val_loss", patience = 5)
	
	callbacks_list = [earlyStopper, checkpointer]
	
	return model.fit(
			train_ds,
			# ~ steps_per_epoch = 1, #to shorten training for testing purposes. I got no gpu qq.
			callbacks = callbacks_list,
			epochs = numEpochs,
			validation_data = val_ds)


def saveGraphs(model, myHistory, test_ds, outputFolder):
	evalLoss, evalAccuracy = model.evaluate(test_ds)

	plt.clf()
	accuracy = myHistory.history['accuracy']
	val_accuracy = myHistory.history["val_accuracy"]
	
	epochs = range(1, len(accuracy) + 1)
	accCap = round(evalAccuracy, 4)
	captionTextAcc = "Accuracy on test data: {}".format(accCap)
	plt.figtext(0.5, 0.01, captionTextAcc, wrap=True, horizontalalignment='center', fontsize=12)
	plt.plot(epochs, accuracy, "o", label="Training accuracy")
	plt.plot(epochs, val_accuracy, "^", label="Validation accuracy")
	plt.title("Model Accuracy vs Epochs")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend()
	plt.savefig(os.path.join(outputFolder, "trainvalacc.png"))

	plt.clf()
	
	loss = myHistory.history["loss"]
	val_loss = myHistory.history["val_loss"]
	
	lossCap = round(evalLoss, 4)
	captionTextLoss = "Loss on test data: {}".format(lossCap)
	plt.figtext(0.5, 0.01, captionTextLoss, wrap=True, horizontalalignment='center', fontsize=12)
	plt.plot(epochs, loss, "o", label="Training loss")
	plt.plot(epochs, val_loss, "^", label="Validation loss")
	plt.title("Training and validation loss vs Epochs")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.legend()
	plt.savefig(os.path.join(outputFolder, "trainvalloss.png"))
	plt.clf()
	
	return captionTextAcc, captionTextLoss


def getDatasets(trainDir, valDir, testDir):
	train = tf.data.experimental.load(trainDir)
	val = tf.data.experimental.load(valDir)
	test = tf.data.experimental.load(testDir)
	
	return train, val, test


# Prints first nine images from the first batch of the dataset.
# It's random as long as you shuffle the dataset! ;)
def printSample(in_ds):
	plt.figure(figsize=(10, 10))
	for img, label in in_ds.take(1):
		# ~ for i in tqdm.tqdm(range(9)):
		for i in tqdm(range(9)):
			ax = plt.subplot(3, 3, i + 1)
			myImg = np.asarray(img)
			plt.imshow(np.asarray(myImg[i]), cmap="gray")
			plt.title( CLASS_NAMES_LIST_STR[ np.asarray(label[i]) ]  )
			plt.axis("off")
		plt.show()
	plt.clf()


# Extract the labels from the tensorflow dataset structure.
def extractLabels(in_ds):
	print("Trying to get list out of test dataset...")
	lablist = []
	for batch in tqdm(in_ds):
		lablist.extend( np.asarray(batch[1]) )
	
	return np.asarray(lablist)
	

def printStringToFile(fileName, textString, openMode):
	with open(fileName, openMode) as myFile:
		for character in textString:
			myFile.write(character)


def printLabelStuffToFile(predictedScores, originalLabels, predictedLabels, outputFolder):
	with open(os.path.join(outputFolder, "predictionlists.txt"), "w") as outFile:
		for i in range(len(predictedScores)):
			thisScores = predictedScores[i]
			bobcatScore = str(round(thisScores[CLASS_BOBCAT], 4))
			coyoteScore = str(round(thisScores[CLASS_COYOTE], 4))
			deerScore = str(round(thisScores[CLASS_DEER], 4))
			elkScore = str(round(thisScores[CLASS_ELK], 4))
			humanScore = str(round(thisScores[CLASS_HUMAN], 4))
			notScore = str(round(thisScores[CLASS_NOT_INTERESTING], 4))
			raccoonScore = str(round(thisScores[CLASS_RACCOON], 4))
			weaselScore = str(round(thisScores[CLASS_WEASEL], 4))
			
			thisString = \
			"predicted scores: [" + bobcatScore + ", " + coyoteScore + ", " + deerScore + ", " + elkScore + ", " + humanScore + ", " + notScore + ", " + raccoonScore + weaselScore + "]" \
			+ "\tactual label " + str(originalLabels[i]) \
			+ "\tpredicted label" + str(predictedLabels[i]) + "\n"
			outFile.write(thisString)	


def getPredictedLabels(testScores):
	outList = []
	for score in testScores:
		"""if score[CLASS_INTERESTING] >= score[CLASS_NOT_INTERESTING]:
			outList.append(CLASS_INTERESTING)
		else:
			outList.append(CLASS_NOT_INTERESTING)"""
   
		thisMax = max(score)
		maxIndex = np.where(score == thisMax)
		outList.append(maxIndex[0][0])
	
 
	return np.asarray(outList)
			

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))





















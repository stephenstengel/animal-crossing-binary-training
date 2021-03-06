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
import math
import gc

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from models import inceptionResNetModel
from keras import callbacks

print("Done!")

TEST_PRINTING = False

IMG_WIDTH = 400
IMG_HEIGHT = 300
IMG_CHANNELS = 3

IMG_SHAPE_TUPPLE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

BATCH_SIZE = 32	# This is also set in the image loader. They must match.
EPOCHS = 50
PATIENCE = 2
REPEATS = 5

# class number is 7 for combined deer elk
# class number is 8 for all classes
CLASS_NUMBER = 7

# how to get programatically? 
MY_PYTHON_STRING = "python"
# ~ MY_PYTHON_STRING = "python3"
# ~ MY_PYTHON_STRING = "py"

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
CLASS_DEER_ELK = -1

# if combined deer elk
if CLASS_NUMBER == 7:
	CLASS_DEER_ELK = 2
	CLASS_HUMAN = 3
	CLASS_NOT_INTERESTING = 4
	CLASS_RACCOON = 5
	CLASS_WEASEL = 6

CLASS_BOBCAT_STRING = "bobcat"
CLASS_COYOTE_STRING = "coyote"
CLASS_DEER_STRING = "deer"
CLASS_ELK_STRING = "elk"
CLASS_DEER_ELK_STRING = "deer-elk"
CLASS_HUMAN_STRING = "human"
CLASS_RACCOON_STRING = "raccoon"
CLASS_WEASEL_STRING = "weasel"
CLASS_NOT_INTERESTING_STRING = "not"

CLASS_NAMES_LIST_INT = [CLASS_BOBCAT, CLASS_COYOTE, CLASS_DEER, CLASS_ELK, CLASS_HUMAN, CLASS_NOT_INTERESTING, CLASS_RACCOON, CLASS_WEASEL]
CLASS_NAMES_LIST_STR = [CLASS_BOBCAT_STRING, CLASS_COYOTE_STRING, CLASS_DEER_STRING, CLASS_ELK_STRING, CLASS_HUMAN_STRING, CLASS_NOT_INTERESTING_STRING, CLASS_RACCOON_STRING, CLASS_WEASEL_STRING]

# if combined deer-elk
if CLASS_NUMBER == 7:
	CLASS_NAMES_LIST_INT = [CLASS_BOBCAT, CLASS_COYOTE, CLASS_DEER_ELK, CLASS_HUMAN, CLASS_NOT_INTERESTING, CLASS_RACCOON, CLASS_WEASEL]
	CLASS_NAMES_LIST_STR = [CLASS_BOBCAT_STRING, CLASS_COYOTE_STRING, CLASS_DEER_ELK_STRING, CLASS_HUMAN_STRING, CLASS_NOT_INTERESTING_STRING, CLASS_RACCOON_STRING, CLASS_WEASEL_STRING]


def main(args):
	#base folder for this run
	ts = time.localtime()
	timeStr = "./%d-%d-%d-%d-%d-%d/" % (ts.tm_year, ts.tm_mon, ts.tm_mday, ts.tm_hour, ts.tm_min, ts.tm_sec)
	timeStr = os.path.normpath(timeStr)
	
	# Folders to save model tests
	inceptionResNetFolder = os.path.join(timeStr, "inceptionResNet")
	modelBaseFolders = [inceptionResNetFolder]
	makeDirectories(modelBaseFolders)
	
	imgShape = IMG_SHAPE_TUPPLE
	classNumber = CLASS_NUMBER
	batchSize = BATCH_SIZE
	numEpochs = EPOCHS
	numPatience = PATIENCE
	
	#these contain the functions to create the models, NOT the models themselves.
	modelList = [inceptionResNetModel]

	#This loop can be segmented further. We could also keep track of the
	#best accuracy from each type of model. Then printout which model
	#gave the best accuracy overall and say where the model is saved.
	for i in range(len(modelList)):
		overallBestAcc = -math.inf
		overallBestModel = None
		overallBestFolder = ""
		overallBestCheckpointFolder = ""
		eachModelAcc = []
		
		thisAcc, thisModel, thisFolder, thisCheckpointFolder = \
				runManyTests(
						modelBaseFolders[i], REPEATS, modelList[i], \
						numEpochs, numPatience, imgShape, classNumber, batchSize, \
						LOADER_DIRECTORY, overallBestCheckpointFolder)
		eachModelAcc.append(thisAcc)
		if thisAcc > overallBestAcc:
			overallBestAcc = thisAcc
			del overallBestModel
			overallBestModel = thisModel
			overallBestFolder = thisFolder
			deleteDirectories([overallBestCheckpointFolder])
			overallBestCheckpointFolder = thisCheckpointFolder
		else:
			del thisModel
			deleteDirectories([thisCheckpointFolder])
		gc.collect()
	
	
	outString = "The best accuracies among the models..." + "\n"
	for thingy in eachModelAcc:
		outString += str(round(thingy, 4)) + "\n"
	outString += "The overall best saved model is in folder: " + overallBestFolder + "\n"
	outString += "It has an accuracy of: " + str(round(overallBestAcc, 4)) + "\n"
	print(outString)
	printStringToFile(os.path.join(timeStr, "overall-output.txt") , outString, "w")
		
		
	print("A winner is YOU!")

	return 0


def runManyTests(thisBaseOutFolder, numRepeats, inputModel, numEpochs, numPatience, imgShapeTupple, classNumber, batchSize, loaderScriptDirectory, overallBestCheckpointFolder):
	saveCopyOfSourceCode(thisBaseOutFolder)
	
	theBestAccuracy = -math.inf
	theBestModel = None
	theBestSavedModelFolder = "" #might not need this if I use the lists.
	#actually if we save to disk each time we can save ram.
	theBestCheckpointFolder = overallBestCheckpointFolder
	
	eachTestAcc = []
	eachTestCM = []
	
	for jay in range(numRepeats):
		reloadImageDatasets(loaderScriptDirectory, "load-dataset.py") # this function could be replaced with a shuffle function. If we had one big dataset file, we could shuffle that instead of reloading the images every time. But this works.
		train_ds, val_ds, test_ds = getDatasets(TRAIN_DIRECTORY, VAL_DIRECTORY, TEST_DIRECTORY)
  
		if TEST_PRINTING:
			printSample(test_ds)
  
		thisInputModel = inputModel(imgShapeTupple, classNumber)
		
		thisTestAcc, thisOutModel, thisOutputFolder, thisCheckpointFolder, thisCM = runOneTest( \
				thisInputModel, os.path.join(thisBaseOutFolder, str(jay)), \
				train_ds, val_ds, test_ds, \
				numEpochs, numPatience, imgShapeTupple, \
				batchSize)
		
		eachTestAcc.append(thisTestAcc)
		eachTestCM.append(thisCM)
		
		if thisTestAcc > theBestAccuracy:
			theBestAccuracy = thisTestAcc
			theRunWithTheBestAccuracy = jay
			del theBestModel
			theBestModel = thisOutModel
			theBestSavedModelFolder = thisOutputFolder
			deleteDirectories([theBestCheckpointFolder])
			theBestCheckpointFolder = thisCheckpointFolder
		else:
			del thisInputModel #To save a bit of ram faster.
			deleteDirectories([thisCheckpointFolder])
		gc.collect()
	
	# average confusion matrix
	eachTestCM = np.array(eachTestCM)
	avgCM = np.zeros((CLASS_NUMBER, CLASS_NUMBER))
	for cm in eachTestCM:
		avgCM += cm
    
	for i in range(len(avgCM)):
		avgCM[i] = [round((x / numRepeats), 1) for x in avgCM[i]]
 
	cf_plot = ConfusionMatrixDisplay(confusion_matrix=avgCM, display_labels=CLASS_NAMES_LIST_STR)
	cf_plot.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format='')
	plt.tight_layout()
	plt.savefig(os.path.join(thisBaseOutFolder, "average_confusion_matrix.png"))
	plt.clf()
 
	# get accurary per class
	avgClassAcc = np.around(avgCM.diagonal() / avgCM.sum(axis=1), decimals=4)
	
	avgAcc = 0
	outString = "The accuracies for this run..." + "\n"
	for thingy in eachTestAcc:
		outString += str(round(thingy, 4)) + "\n"
		avgAcc += thingy
	avgAcc = round((avgAcc / numRepeats), 4)

	outString += "The best saved model is in folder: " + theBestSavedModelFolder + "\n"
	outString += "It has an accuracy of: " + str(round(theBestAccuracy, 4)) + "\n\n"
	outString += "Averages for " + str(numRepeats) + " tests:\nThe average accuracy was: " + str(avgAcc) + "\n"
	outString += "The average class accuracies were:\n"
	for cName in CLASS_NAMES_LIST_STR:
		outString += cName + ", "
	outString += "\n" + str(avgClassAcc) + "\n"
	print(outString)
	printStringToFile(os.path.join(thisBaseOutFolder, "repeats-output.txt") , outString, "w")
	
	return theBestAccuracy, theBestModel, theBestSavedModelFolder, theBestCheckpointFolder


def runOneTest(thisModel, thisOutputFolder, train_ds, val_ds, test_ds, numEpochs, numPatience, imgShapeTupple, batchSize):
	thisModel.summary()
	print("Training model: " + thisOutputFolder)
	thisCheckpointFolder = os.path.abspath(os.path.join(thisOutputFolder, "checkpoint"))
	thisMissclassifiedFolder = os.path.join(thisOutputFolder, "misclassifed-images")
	foldersForThisModel = [thisOutputFolder, thisCheckpointFolder, thisMissclassifiedFolder]
	makeDirectories(foldersForThisModel)
	
	myHistory = trainModel(thisModel, train_ds, val_ds, thisCheckpointFolder, numEpochs, numPatience)
	print("Creating graphs of training history...")
	#thisTestAcc is the same as strAcc but in unrounded float form.
	strAcc, strLoss, thisTestAcc = saveGraphs(thisModel, myHistory, test_ds, thisOutputFolder)
 
	CMandCR, cm= evaluateLabels(test_ds, thisModel, thisOutputFolder, thisMissclassifiedFolder, batchSize)

	#workin on this.
	stringToPrint = "Epochs: " + str(numEpochs) + "\n"
	stringToPrint += "Image Shape: " + str(imgShapeTupple) + "\n\n"
	stringToPrint += CMandCR
	stringToPrint += "Accuracy and loss according to tensorflow model.evaluate():\n"
	stringToPrint += strAcc + "\n"
	stringToPrint += strLoss + "\n"
	
	statFileName = os.path.join(thisOutputFolder, "stats.txt")
	printStringToFile(statFileName, stringToPrint, "w")
	print(stringToPrint)
	
	return thisTestAcc, thisModel, thisOutputFolder, thisCheckpointFolder, cm
	

#Reload the images from the dataset so that you can run another test with randomized images.
def reloadImageDatasets(loaderPath, scriptName):
	#save current directory
	startDirectory = os.getcwd()
	os.chdir(loaderPath)
	
	loaderPID = None
	
	os.system(MY_PYTHON_STRING + " " + scriptName)
	
	# ~ loaderPID = subprocess.Popen([MY_PYTHON_STRING, scriptName])
	# ~ if loaderPID is not None:
		# ~ loaderPID.wait()
	# ~ else:
		# ~ print("MASSIVE ERROR LOL!")
	
	os.chdir(startDirectory)


#Runs a system command. Input is the string that would run on linux or inside wsl.
def runSystemCommand(inputString):
	if sys.platform.startswith("win"):
		os.system("wsl " + inputString)
	elif sys.platform.startswith("linux"):
		os.system(inputString)
	else:
		print("MASSIVE ERROR LOL!")
		exit(-4)	

#save copy of source code.
def saveCopyOfSourceCode(thisOutputFolder):
	thisFileName = os.path.basename(__file__)
	try:
		shutil.copy(thisFileName, os.path.join(thisOutputFolder, "copy-" + thisFileName))
	except:
		print("Failed to make a copy of the source code!")


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
	for i in range(len(CLASS_NAMES_LIST_STR)):
		outString += CLASS_NAMES_LIST_STR[i] + "(" + str(i) + "), "
	
	cm = confusion_matrix(actual_test_labels, p_test_labels)
	cm_report = classification_report(actual_test_labels, p_test_labels, digits=4)
	cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES_LIST_STR)
	cm_plot.plot(cmap=plt.cm.Blues, xticks_rotation=45)
	plt.tight_layout()
	plt.savefig(os.path.join(outputFolder, "confusion_matrix.png"))
	plt.clf()
	
 
	outString += "\n" + str(cm) + "\n" + cm_report + "\n"	
	
	#Make a pretty chart of these images?
	
	return outString, cm


# Saves all missclassified images
def saveMisclassified(dataset, labels, predicted, missClassifiedFolder, batchSize):
	cnt = 0
	for img, _ in dataset.take(-1):
		for i in range(batchSize):
			if labels[cnt] != predicted[cnt]:
				myImg = np.asarray(img)
				thisActualName = CLASS_NAMES_LIST_STR[labels[cnt]]
				thisPredictedName = CLASS_NAMES_LIST_STR[predicted[cnt]]
				thisFileString = \
						"actual_" + thisActualName \
						+ "_predicted_" +  thisPredictedName \
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
def trainModel(model, train_ds, val_ds, checkpointFolder, numEpochs, numPatience):
	checkpointer = callbacks.ModelCheckpoint(
		filepath = checkpointFolder,
		monitor = "accuracy",
		save_best_only = True,
		mode = "max")
	
	earlyStopper = callbacks.EarlyStopping( \
			monitor="val_accuracy", \
			mode = "max",
			patience = numPatience, \
			restore_best_weights = True)
	
	callbacks_list = [earlyStopper, checkpointer]
	
	return model.fit(
			train_ds,
			# ~ steps_per_epoch = 1, #to shorten training for testing purposes. I got no gpu qq.
			callbacks = callbacks_list,
			epochs = numEpochs,
			validation_data = val_ds)


# Returns caption strings for the graphs of the accuracy and loss
# also returns the accuracy of the model as applied to the test dataset.
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
	
	return captionTextAcc, captionTextLoss, evalAccuracy


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
			thisString = "predicted scores: [" 
			for animalClass in CLASS_NAMES_LIST_INT:
				thisString += str(round(thisScores[animalClass], 4))
				if len(thisScores) - 1 != animalClass:
					thisString += ", "
			thisString += "]" \
					+ "\tactual label " + str(originalLabels[i]) \
					+ "\tpredicted label" + str(predictedLabels[i]) \
					+ "\n"
			outFile.write(thisString)	


def getPredictedLabels(testScores):
	outList = []
	for score in testScores:
		outList.append(np.argmax(score))
	
 
	return np.asarray(outList)
			

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))

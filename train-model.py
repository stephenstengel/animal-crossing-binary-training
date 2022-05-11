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

from tqdm import tqdm

from models import createHarlowModel, simpleModel, inceptionV3Model
from keras import callbacks
from keras import backend

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
IMG_CHANNELS = 3

IMG_SHAPE_TUPPLE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


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
	modelList = [simpleModel(shape), createHarlowModel(shape), inceptionV3Model(shape)]

	# This for loop can be compartmentalized into helper functions.
	# There will be one wrapper function to perform k-folds
	# something like performExperiment -> performKfolds -> contents of this for loop.
	for i in range(len(modelList)):
		thisModel = modelList[i]
		thisModel.summary()
		thisOutputFolder = modelBaseFolders[i]
		print("Training model: " + thisOutputFolder)
		thisCheckpointFolder = os.path.join(thisOutputFolder, "checkpoint")
		foldersForThisModel = [thisOutputFolder, thisCheckpointFolder]
		makeDirectories(foldersForThisModel)
		
		#save copy of source code that created the output
		os.system("cp train-model.py   " + os.path.join(thisOutputFolder, "train-model.py"))
		
		myHistory = trainModel(thisModel, train_ds, val_ds, thisCheckpointFolder)
		print("Creating graphs of training history...")
		strAcc, strLoss = saveGraphs(thisModel, myHistory, test_ds, thisOutputFolder)
		
		#workin on this.
		stringToPrint = evaluateLabels(test_ds, thisModel, thisOutputFolder)
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
def evaluateLabels(test_ds, model, outputFolder):
	print("Getting predictions of test data...")
	testScores = model.predict(test_ds, verbose = True)
	actual_test_labels = extractLabels(test_ds)
	
	#Get the list of class predictions from the probability scores.
	p_test_labels = getPredictedLabels(testScores)
	
	printLabelStuffToFile(testScores, actual_test_labels, p_test_labels, outputFolder) # debug function
	
	#Calculate TPR, FPR, TNR, FNR
	outString = ""
	tp_sum = getTPsum(actual_test_labels, p_test_labels)
	outString += "truePos: " + str(tp_sum) + "\n"
	tn_sum = getTNsum(actual_test_labels, p_test_labels)
	outString += "true negative: " + str(tn_sum) + "\n"
	fp_sum = getFPsum(actual_test_labels, p_test_labels)
	outString += "false pos: " + str(fp_sum) + "\n"
	fn_sum = getFNsum(actual_test_labels, p_test_labels)
	outString += "false negative: " + str(fn_sum) + "\n"
	
	accuracy = getAcc(tp_sum, tn_sum, fp_sum, fn_sum)
	outString += "accuracy: " + str(accuracy) + "\n"
	err = getErrRate(tp_sum, tn_sum, fp_sum, fn_sum)
	outString += "error rate: " + str(err) + "\n"
	
	tpr = getTPR(tp_sum, fn_sum)
	outString += "True Positive Rate: " + str(tpr) + "\n"
	
	tNr = getTNR(tn_sum, fp_sum)
	outString += "True Negative Rate: " + str(tNr) + "\n"
	
	precision = getPrecision(tp_sum, fp_sum)
	outString += "Precision: " + str(precision) + "\n"
	
	
	#Save the false positive, false negative images into folders.
	
	#Make a pretty chart of these images?
	
	return outString


def getAcc(tp, tn, fp, fn):
	top = tp + tn
	bottom = tp + fp + tn + fn
	
	return top / bottom

def getErrRate(tp, tn, fp, fn):
	return 1 - getAcc(tp, tn, fp, fn)


# Also known as Sensitivity, recall, and hit rate.
def getTPR(tp, fn):
	return tp / (tp + fn)
	

# Also known as Specificity and selectivity
def getTNR(tn, fp):
	return tn / (tn + fp)


# Also known as positive predictive value
def getPrecision(truePos, falsePos):
	return truePos / (truePos + falsePos)
	

# have to think how to do the mask to go very fast.
# i'll just do a loop for now
# I think a lambda function thing would work.
def getTPsum(actual_test_labels, p_test_labels):
	sumList = []
	for i in range(len(actual_test_labels)):
		if (actual_test_labels[i] == CLASS_BOBCAT) and (actual_test_labels[i] == p_test_labels[i]):
			sumList.append(0)
		elif (actual_test_labels[i] == CLASS_COYOTE) and (actual_test_labels[i] == p_test_labels[i]):
				sumList.append(1)
		elif (actual_test_labels[i] == CLASS_DEER) and (actual_test_labels[i] == p_test_labels[i]):
				sumList.append(2)
		elif (actual_test_labels[i] == CLASS_ELK) and (actual_test_labels[i] == p_test_labels[i]):
				sumList.append(3)
		elif (actual_test_labels[i] == CLASS_HUMAN) and (actual_test_labels[i] == p_test_labels[i]):
				sumList.append(4)
		elif (actual_test_labels[i] == CLASS_NOT_INTERESTING) and (actual_test_labels[i] == p_test_labels[i]):
				sumList.append(5)
		elif (actual_test_labels[i] == CLASS_RACCOON) and (actual_test_labels[i] == p_test_labels[i]):
				sumList.append(6)
		else:
			sumList.append(7)
	
	sumArr = np.asarray(sumList)
	
	return np.asarray(backend.sum(sumArr))


def getTNsum(actual_test_labels, p_test_labels):
	sumList = []
	for i in range(len(actual_test_labels)):
		if (actual_test_labels[i] == CLASS_NOT_INTERESTING) and (actual_test_labels[i] == p_test_labels[i]):
			sumList.append(1)
		else:
			sumList.append(0)
	
	sumArr = np.asarray(sumList)
	
	return np.asarray(backend.sum(sumArr))
	
	


def getFPsum(actual_test_labels, p_test_labels):
	sumList = []
	for i in range(len(actual_test_labels)):
		if (actual_test_labels[i] == CLASS_NOT_INTERESTING) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(1)
		else:
			sumList.append(0)
	
	sumArr = np.asarray(sumList)
	
	return np.asarray(backend.sum(sumArr))


def getFNsum(actual_test_labels, p_test_labels):
	sumList = []
	for i in range(len(actual_test_labels)):
		if (actual_test_labels[i] == CLASS_BOBCAT) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(0)
		elif (actual_test_labels[i] == CLASS_COYOTE) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(1)
		elif (actual_test_labels[i] == CLASS_DEER) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(2)
		elif (actual_test_labels[i] == CLASS_ELK) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(3)
		elif (actual_test_labels[i] == CLASS_HUMAN) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(4)
		elif (actual_test_labels[i] == CLASS_NOT_INTERESTING) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(5)
		elif (actual_test_labels[i] == CLASS_RACCOON) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(6)
		else:
			sumList.append(7)
	
	sumArr = np.asarray(sumList)
	
	return np.asarray(backend.sum(sumArr))
	
	


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
def trainModel(model, train_ds, val_ds, checkpointFolder):
	checkpointer = callbacks.ModelCheckpoint(
		filepath = checkpointFolder,
		monitor = "accuracy",
		save_best_only = True,
		mode = "max")
	
	earlyStopper = callbacks.EarlyStopping(monitor="accuracy", patience = 10)
	
	callbacks_list = [earlyStopper, checkpointer]
	
	return model.fit(
			train_ds,
			# ~ steps_per_epoch = 1, #to shorten training for testing purposes. I got no gpu qq.
			callbacks = callbacks_list,
			epochs = 15,
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
			"predicted score bobcat, coyote, deer, elk, human, not, raccon, score: [" + bobcatScore + ", " + coyoteScore + ", " + deerScore + ", " + elkScore + ", " + humanScore + ", " + notScore + ", " + raccoonScore + weaselScore + "]" \
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
		outList.append(maxIndex)
	
 
	return np.asarray(outList)
			

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))





















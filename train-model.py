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

from tqdm import tqdm

from models import createHarlowModel
from keras import callbacks
from keras import backend

print("Done!")


TRAIN_DIRECTORY = "../load-dataset-script/dataset/train/"
VAL_DIRECTORY = "../load-dataset-script/dataset/val/"
TEST_DIRECTORY = "../load-dataset-script/dataset/test/"

CLASS_INTERESTING = 0
CLASS_NOT_INTERESTING = 1

CLASS_INTERESTING_STRING = "interesting"
CLASS_NOT_INTERESTING_STRING = "not"

CLASS_NAMES_LIST_INT = [CLASS_INTERESTING, CLASS_NOT_INTERESTING]
CLASS_NAMES_LIST_STR = [CLASS_INTERESTING_STRING, CLASS_NOT_INTERESTING_STRING]

TEST_PRINTING = True

IMG_WIDTH = 100
IMG_HEIGHT = 100
# ~ IMG_WIDTH = 35
# ~ IMG_HEIGHT = 35
IMG_CHANNELS = 1

IMG_SHAPE_TUPPLE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)


def main(args):
	listOfFoldersToDELETE = []
	deleteDirectories(listOfFoldersToDELETE)
	
	# Folders to save model tests
	harlowFolder = "./harlow/"
	modelBaseFolders = [harlowFolder] #Same order as the modelList below!
	
	listOfFoldersToMake = [harlowFolder]
	makeDirectories(listOfFoldersToMake)
	
	
	# train_ds is for training the model.
	# val_ds is for validation during training.
	# test_ds is a dataset of unmodified images for testing the model after training.
	train_ds, val_ds, test_ds = getDatasets(TRAIN_DIRECTORY, VAL_DIRECTORY, TEST_DIRECTORY)
	
	
	if TEST_PRINTING:
		printSample(test_ds)
	
	shape = IMG_SHAPE_TUPPLE
	modelList = [createHarlowModel(shape)]
	
	for model in modelList:
		model.summary()
	
	for i in range(len(modelList)):
		print("Training model...")
		thisOutputFolder = modelBaseFolders[i]
		thisCheckpointFolder = thisOutputFolder + "checkpoint/"
		foldersForThisModel = [thisOutputFolder, thisCheckpointFolder]
		makeDirectories(foldersForThisModel)
		
		myHistory = trainModel(model, train_ds, val_ds, thisCheckpointFolder)
		print("Creating graphs of training history...")
		strAcc, strLoss = saveGraphs(model, myHistory, test_ds, thisOutputFolder)
		
		#workin on this.
		stringToPrint = evaluateLabels(test_ds, model, thisOutputFolder)
		stringToPrint += "Accuracy and loss according to tensorflow model.evaluate():\n"
		stringToPrint += strAcc + "\n"
		stringToPrint += strLoss + "\n"
		
		statFileName = thisOutputFolder + "stats.txt"
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
		if (actual_test_labels[i] == CLASS_INTERESTING) and (actual_test_labels[i] == p_test_labels[i]):
			sumList.append(1)
		else:
			sumList.append(0)
	
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
		if (actual_test_labels[i] == CLASS_INTERESTING) and (actual_test_labels[i] != p_test_labels[i]):
			sumList.append(1)
		else:
			sumList.append(0)
	
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
	callbacks_list = [checkpointer]
	
	return model.fit(
			train_ds,
			# ~ steps_per_epoch = 1, #to shorten training for testing purposes. I got no gpu qq.
			callbacks = callbacks_list,
			epochs = 150,
			# ~ epochs = 5,
			# ~ epochs = 1,
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
	plt.savefig(outputFolder + "trainvalacc.png")

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
	plt.savefig(outputFolder + "trainvalloss.png")
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
	with open(outputFolder + "predictionlists.txt", "w") as outFile:
		for i in range(len(predictedScores)):
			thisScores = predictedScores[i]
			intScore = str(round(thisScores[CLASS_INTERESTING], 4))
			notScore = str(round(thisScores[CLASS_NOT_INTERESTING], 4))
			
			thisString = \
			"predicted score int,not: [" + intScore + ", " + notScore + "]" \
			+ "\tactual label " + str(originalLabels[i]) \
			+ "\tpredicted label" + str(predictedLabels[i]) + "\n"
			outFile.write(thisString)	

def getPredictedLabels(testScores):
	outList = []
	for score in testScores:
		if score[CLASS_INTERESTING] >= score[CLASS_NOT_INTERESTING]:
			outList.append(CLASS_INTERESTING)
		else:
			outList.append(CLASS_NOT_INTERESTING)
	
	return np.asarray(outList)
			


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))





















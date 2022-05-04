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

print("Done!")


TRAIN_DIRECTORY = "../load-dataset-script/dataset/train/"
VAL_DIRECTORY = "../load-dataset-script/dataset/val/"
TEST_DIRECTORY = "../load-dataset-script/dataset/test/"

CHECKPOINT_FOLDER = "./checkpoint/"

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
	listOfFolders = [CHECKPOINT_FOLDER]
	makeDirectories(listOfFolders)
	
	
	# train_ds is for training the model.
	# val_ds is for validation during training.
	# test_ds is a dataset of unmodified images for testing the model after training.
	train_ds, val_ds, test_ds = getDatasets(TRAIN_DIRECTORY, VAL_DIRECTORY, TEST_DIRECTORY)
	
	extractTest(test_ds)
	
	if TEST_PRINTING:
		printSample(test_ds)
	
	shape = IMG_SHAPE_TUPPLE
	modelList = [createHarlowModel(shape)]
	
	for model in modelList:
		model.summary()
	
	
	#make output folders for each model
	for model in modelList:
		myHistory = trainModel(model, train_ds, val_ds, CHECKPOINT_FOLDER)
		testScores = model.predict(test_ds, verbose = True)
		evalLoss, evalAccuracy = model.evaluate(test_ds)
		examineResults(model, myHistory, testScores, evalLoss, evalAccuracy)

	return 0


# Creates the necessary directories.
def makeDirectories(listOfFoldersToCreate):
	if os.path.isdir(CHECKPOINT_FOLDER):
		shutil.rmtree(CHECKPOINT_FOLDER, ignore_errors = True)
	
	for folder in listOfFoldersToCreate:
		if not os.path.isdir(folder):
			os.makedirs(folder)


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
			steps_per_epoch = 20, #to shorten training for testing purposes. I got no gpu qq.
			callbacks = callbacks_list,
			epochs = 5,
			validation_data = val_ds)


def examineResults(model, myHistory, testScores, evalLoss, evalAccuracy):
	accuracy = myHistory.history['accuracy']
	epochs = range(1, len(accuracy) + 1)
	plt.plot(epochs, accuracy, "o", label="Training accuracy")
	plt.title("Model Accuracy vs Epochs")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.legend()
	plt.savefig("train-history.png")
	plt.clf()
	
	print("\nEvaluation on Test Data: Loss = {}, accuracy = {}".format(round(evalLoss, 5), round(evalAccuracy, 5)))
	
	print("now scores object from model.predict...")
	print("hehe: %.2f%%" % (testScores[1]))
	print("%s: %.2f%%" % (model.metrics_names[1], testScores[1]))
	



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


def extractTest(test_ds):
	print("Trying to get list out of test dataset...")
	lablist = []
	imglist = []
	for batch in tqdm(test_ds):
		imglist.extend( np.asarray(batch[0]) )
		lablist.extend( np.asarray(batch[1]) )
	# ~ imglist = np.asarray(imglist)
	# ~ lablist = np.asarray(lablist)
	print("len imglist: " + str(len(imglist)))
	# ~ print(imglist)
	print("len lablist: " + str(len(lablist)))
	print(lablist)
	
	
	#try to print the images.
	plt.clf()
	images = []
	labels = []
	for batch in test_ds:
		imgArr = np.asarray(batch[0])
		labelArr = np.asarray(batch[1])
		for i in range(len(imgArr)):
			thisImg = imgArr[i]
			# ~ thisImg = img_as_uint(thisImg)
			thisLabel = labelArr[i]
			
			images.append(thisImg)
			labels.append(thisLabel)

	# ~ plt.figure(figsize=(10, 10))
	for i in range(len(images)):
		myImg = images[i]
		myLabel = labels[i]
		plt.imshow(myImg, cmap="gray")
		plt.title( CLASS_NAMES_LIST_STR[ myLabel ]  )
		plt.axis("off")
		plt.show()
		plt.clf()
	plt.clf()
	
	
	sys.exit(-2)


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))





















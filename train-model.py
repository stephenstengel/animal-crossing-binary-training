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

from models import createHarlowModel

print("Done!")


TRAIN_DIRECTORY = "./dataset/train/"
TEST_DIRECTORY = "./dataset/test/"

CLASS_INTERESTING = 0
CLASS_NOT_INTERESTING = 1

CLASS_INTERESTING_STRING = "interesting"
CLASS_NOT_INTERESTING_STRING = "not"

CLASS_NAMES_LIST_INT = [CLASS_INTERESTING, CLASS_NOT_INTERESTING]
CLASS_NAMES_LIST_STR = [CLASS_INTERESTING_STRING, CLASS_NOT_INTERESTING_STRING]

TEST_PRINTING = False



def main(args):
	train_ds, test_ds = getDatasets(TRAIN_DIRECTORY, TEST_DIRECTORY)
	createHarlowModel()
	

	return 0


def getDatasets(trainDir, testDir):
	train = tf.data.experimental.load(trainDir)
	test = tf.data.experimental.load(testDir)
	
	return train, test


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))

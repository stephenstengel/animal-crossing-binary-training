#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  models.py
#  
#  Copyright 2022 Stephen Stengel <stephen.stengel@cwu.edu> and friends
#  

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy

def createHarlowModel(shapeTupple):
	hModel = Sequential(
		[
			Input(shapeTupple),
			Conv2D(32, 3, 1, padding='same', activation='relu'),
			Conv2D(64, 3, 1, padding='same', activation='relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Conv2D(128, 3, 1, padding='same', activation='relu'),
			Conv2D(256, 3, 1, padding='same', activation='relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Flatten(),
			Dense(128, activation='relu'),
			Dense(2, activation='softmax') #Needed for sparse categorical crossentropy
		]
	)
	
	hModel.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
	
	
	return hModel



def simpleModel(shapeTupple):
	model = tf.keras.models.Sequential([
		Input(shapeTupple),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(2, activation = "softmax")
	])
	
	model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
	
	return model

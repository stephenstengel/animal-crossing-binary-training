#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  models.py
#  
#  Copyright 2022 Stephen Stengel <stephen.stengel@cwu.edu> and friends
#  

from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import BinaryCrossentropy

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
			Dense(1)
		]
	)
	
	# ~ hModel.compile(
		# ~ optimizer='adam', # default learning rate is 0.001
		# ~ loss='sparse_categorical_crossentropy',
		# ~ metrics=['accuracy'])
	hModel.compile(
		optimizer='adam', # default learning rate is 0.001
		loss = BinaryCrossentropy(from_logits=True),
		metrics=['accuracy'])
	
	
	return hModel

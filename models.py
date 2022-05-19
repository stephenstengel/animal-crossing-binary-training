#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  models.py
#  
#  Copyright 2022 Stephen Stengel <stephen.stengel@cwu.edu> and friends
#  

import tensorflow as tf
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy

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
			Dense(8, activation='softmax') #Needed for sparse categorical crossentropy
		]
	)
	
	hModel.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		# ~ metrics=['accuracy'])
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name = "accuracy")])
	
	
	return hModel


def inceptionV3Model(shapeTupple):
    base_model = InceptionV3(
		weights='imagenet',
		include_top=False,
		input_shape=shapeTupple
	)
    
    base_model.trainable = False
    
    v3_model = Sequential(
		[
			base_model,
			MaxPooling2D(pool_size=(2, 2), padding='same'),
			Dropout(0.1),
			Flatten(),
			Dense(64, activation='relu'),
			Dense(64, activation='relu'),
			Dense(32, activation='relu'),
			Dense(8, activation='softmax')
		]
	)
    
    v3_model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name = "accuracy")])
    
    return v3_model


def simpleModel(shapeTupple):
	model = tf.keras.models.Sequential([
		Input(shapeTupple),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(8, activation = "softmax")
	])
	
	model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
	
	return model


def mediumModel(shapeTupple):
	model = tf.keras.models.Sequential([
		Input(shapeTupple),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(256, activation='relu'),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(256, activation='relu'),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(8, activation = "softmax")
	])
	
	model.compile(
		optimizer=tf.keras.optimizers.SGD(),
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name = "accuracy")]) #used name ="accuracy" so I don't have to re-write graphs
	
	return model

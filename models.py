#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  models.py
#  
#  Copyright 2022 Stephen Stengel <stephen.stengel@cwu.edu> and friends
#  

import tensorflow as tf
from keras.models import Sequential
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.losses import SparseCategoricalCrossentropy


def inceptionResNetModel(shapeTupple):
    base_model = InceptionResNetV2(
		weights='imagenet',
		include_top=False,
		input_shape=shapeTupple
	)
    
    base_model.trainable = False
    
    incepnet = Sequential(
		[
			base_model,
			GlobalAveragePooling2D(),
			#MaxPooling2D(pool_size=(2, 2), padding='same'),
			#Dropout(0.1),
			#Flatten(),
			#Dense(64, activation='relu'),
			#Dense(64, activation='relu'),
			#Dense(32, activation='relu'),
			Dense(8, activation='softmax')
		]
	)
    
    incepnet.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
    
    return incepnet
 
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
from keras.layers import Dense, GlobalAveragePooling2D
from keras.losses import SparseCategoricalCrossentropy


def inceptionResNetModel(shapeTupple, classNumber):
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
			Dense(classNumber, activation='softmax')
		]
	)
    
    incepnet.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
    
    return incepnet
 
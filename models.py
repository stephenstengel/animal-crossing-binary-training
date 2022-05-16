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
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNet
from keras.applications.efficientnet_v2 import EfficientNetV2M
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import SparseCategoricalCrossentropy

def createHarlowModel(shapeTupple):
	hModel = Sequential(
		[
			Input(shapeTupple),
			Conv2D(32, 3, 1, padding='same', activation='relu'),
			Conv2D(64, 3, 1, padding='same', activation='relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Conv2D(32, 3, 1, padding='same', activation='relu'),
			Conv2D(64, 3, 1, padding='same', activation='relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Flatten(),
			Dense(128, activation='relu'),
			Dense(8, activation='softmax') #Needed for sparse categorical crossentropy
		]
	)
 
	hModel._name = 'Harlow'
	
	hModel.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
	
	
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
		], 
	)
    
    v3_model._name = 'InceptionV3'
    
    v3_model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
    
    return v3_model


def simpleModel(shapeTupple):
	model = tf.keras.models.Sequential([
		Input(shapeTupple),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.1),
		tf.keras.layers.Dense(8, activation = "softmax")
	])
 
	model._name='simple'
	
	model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
	
	return model

def VGG16Model(shapeTupple):
    base_model = VGG16(
		weights='imagenet',
		include_top=False,
		input_shape=shapeTupple
	)
    
    base_model.trainable = False
    
    model = Sequential(
		[
			base_model,
			MaxPooling2D(pool_size=(2, 2), padding='same'),
			Dropout(0.1),
			Flatten(),
			Dense(32, activation='relu'),
			Dense(8, activation='softmax')
		]
	)
    
    model._name = 'VGG16'
    
    model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
    
    return model

def VGG19Model(shapeTupple):
    base_model = VGG19(
		weights='imagenet',
		include_top=False,
		input_shape=shapeTupple
	)
    
    base_model.trainable = False
    
    model = Sequential(
		[
			base_model,
			MaxPooling2D(pool_size=(2, 2), padding='same'),
			Dropout(0.1),
			Flatten(),
			Dense(32, activation='relu'),
			Dense(8, activation='softmax')
		]
	)
    
    model._name = 'VGG19'
    
    model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
    
    return model

# extrememly slow, accuracy and val accuracy was <50% after 5 epochs
def nasNetModel(shapeTupple):
    base_model = NASNet(
		weights='imagenet',
		include_top=False,
		input_shape=shapeTupple
	)
    
    base_model.trainable = False
    
    model = Sequential(
		[
			base_model,
			MaxPooling2D(pool_size=(2, 2), padding='same'),
			Dropout(0.1),
			Flatten(),
			Dense(32, activation='relu'),
			Dense(8, activation='softmax')
		]
	)
    
    model._name = 'NASNet'
    
    model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
    
    return model


def inceptionResNetModel(shapeTupple):
    base_model = InceptionResNetV2(
		weights='imagenet',
		include_top=False,
		input_shape=shapeTupple
	)
    
    base_model.trainable = False
    
    model = Sequential(
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
    
    model._name = 'InceptionResNet'
    
    model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
    
    return model

def xceptionModel(shapeTupple):
    base_model = Xception(
		weights='imagenet',
		include_top=False,
		input_shape=shapeTupple
	)
    
    base_model.trainable = False
    
    model = Sequential(
		[
			base_model,
			MaxPooling2D(pool_size=(2, 2), padding='same'),
			Dropout(0.1),
			Flatten(),
			Dense(32, activation='relu'),
			Dense(8, activation='softmax')
		]
	)
    
    model._name = 'Xception'
    
    model.compile(
		optimizer=tf.keras.optimizers.Adam(), # default learning rate is 0.001
		loss = SparseCategoricalCrossentropy(from_logits=False),
		metrics=['accuracy'])
    
    return model
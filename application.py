import os
from os import listdir
import PIL
import numpy


from numpy.lib.function_base import average


from numpy import zeros
from numpy import asarray

from mrcnn.config import Config

from mrcnn.model import MaskRCNN

from skimage.draw import polygon2mask
from skimage.io import imread

from datetime import datetime



from io import BytesIO
from mrcnn.utils import extract_bboxes
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.backend import clear_session
import json
from flask import Flask, flash, request,jsonify, redirect, url_for
from werkzeug.utils import secure_filename

from skimage.io import imread
from mrcnn.model import mold_image

import tensorflow as tf
import sys
import base64
import io
from PIL import Image
global _model
global _graph
global cfg
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)


MODEL_NAME = "mask_rcnn_hq"

application=Flask(__name__)
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "floorPlan_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 3
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
@application.before_first_request
def load_model():
	global cfg
	global _model
	model_folder_path = os.path.abspath("./") + "/mrcnn"
	weights_path=os.path.abspath("./")+'/weights/mask_rcnn_kangaroo_cfg_0002.h5'
	cfg=PredictionConfig()
	print("=========before loading======")
	_model = MaskRCNN(mode='inference', model_dir=model_folder_path,config=cfg)
	print("==========model Loaded successfully!==========")
	model_path = './weights/mask_rcnn_kangaroo_cfg_0002.h5'
	print("=====before loading weights==========")
	_model.load_weights(weights_path, by_name=True)
	print("==========weights Loaded successfully!==========")
	global _graph
	_graph = tf.get_default_graph()
print("==========before calling load model=======")

def myImageLoader(imageInput):
	image =  numpy.asarray(imageInput)
	
	
	h,w,c=image.shape 
	if image.ndim != 3:
		image = skimage.color.gray2rgb(image)
		if image.shape[-1] == 4:
			image = image[..., :3]
	return image,w,h

def getClassNames(classIds):
	result=list()
	for classid in classIds:
		data={}
		if classid==1:
			data['name']='wall'
		if classid==2:
			data['name']='window'
		if classid==3:
			data['name']='door'
		result.append(data)	

	return result				
def normalizePoints(bbx,classNames):
	normalizingX=1
	normalizingY=1
	result=list()
	doorCount=0
	index=-1
	doorDifference=0
	for bb in bbx:
		index=index+1
		if(classNames[index]==3):
			doorCount=doorCount+1
			if(abs(bb[3]-bb[1])>abs(bb[2]-bb[0])):
				doorDifference=doorDifference+abs(bb[3]-bb[1])
			else:
				doorDifference=doorDifference+abs(bb[2]-bb[0])


		result.append([bb[0]*normalizingY,bb[1]*normalizingX,bb[2]*normalizingY,bb[3]*normalizingX])
	return result,(doorDifference/doorCount)	
		

def turnSubArraysToJson(objectsArr):
	result=list()
	for obj in objectsArr:
		data={}
		data['x1']=obj[1]
		data['y1']=obj[0]
		data['x2']=obj[3]
		data['y2']=obj[2]
		result.append(data)
	return result






@application.route('/',methods=['POST'])
def prediction():
	global cfg
	imagefile = PIL.Image.open(request.files['image'].stream)
	image,w,h=myImageLoader(imagefile)
	scaled_image = mold_image(image, cfg)
	sample = expand_dims(scaled_image, 0)

	global _model
	global _graph
	with _graph.as_default():
		r = _model.detect(sample, verbose=0)[0]
	
	#output_data = model_api(imagefile)
	
	data={}
	bbx=r['rois'].tolist()
	temp,averageDoor=normalizePoints(bbx,r['class_ids'])
	temp=turnSubArraysToJson(temp)
	data['points']=temp
	data['classes']=getClassNames(r['class_ids'])
	data['Width']=w
	data['Height']=h
	data['averageDoor']=averageDoor
	
	return jsonify(data)
	
	
	
	
	
		
	
    
if __name__ =='__main__':
    print("++++++++++starting server++++++++++++++")
	
    application.debug=True
    application.run()
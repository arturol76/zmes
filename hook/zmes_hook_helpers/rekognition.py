import boto3
import io
from PIL import Image

import numpy as np
import zmes_hook_helpers.common_params as g
import zmes_hook_helpers.log as log
import sys
import cv2
from imutils.object_detection import non_max_suppression


# Class to handle HOG based detection

class Rekognition:
    def __init__(self):
        #params from config
        self.aws_region = g.config['aws_region']
        self.aws_access_key_id = g.config['aws_access_key_id']
        self.aws_access_key_secret = g.config['aws_access_key_secret']
        
        self.client=boto3.client(
            'rekognition',
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_access_key_secret
        )

        self.rekognition_labels= {
            "Person": "person",
            "Car": "car"
        }
                
        g.logger.debug('Initializing REKOGNITION')

    def get_classes(self):
        return ['person']

    #person|car|motorbike|bus|truck
    def convert_label(self, label):
        return self.rekognition_labels[label]

    def detect(self, image):

        imgHeight, imgWidth = image.shape[:2]

        pil_img = Image.fromarray(image) # convert opencv frame (with type()==numpy) into PIL Image
        stream = io.BytesIO()
        pil_img.save(stream, format='JPEG') # convert PIL Image to Bytes
        bin_img = stream.getvalue()
        
        g.logger.info("[REKOGNITION] request via boto3...")
        response = self.client.detect_labels(Image={'Bytes': bin_img})
        print(response)
        g.logger.info("[REKOGNITION] ...response received")
        
        bbox = []
        labels = []
        #classes = []
        conf = []

        for label in response['Labels']:
            for instance in label['Instances']:
                if label['Name'] in self.rekognition_labels:
                    
                    labels.append(self.convert_label(label['Name']))
                    conf.append(float(label['Confidence'])/100)

                    #extracting bounding box coordinates
                    left = int(imgWidth * instance['BoundingBox']['Left'])
                    top = int(imgHeight * instance['BoundingBox']['Top'])
                    width = int(imgWidth * instance['BoundingBox']['Width'])
                    height = int(imgHeight * instance['BoundingBox']['Height'])

                    x1 = left
                    y1 = top
                    x2 = left+width
                    y2 = top+height

                    #bbox
                    bbox.append([x1, y1, x2, y2])

        return bbox, labels, conf

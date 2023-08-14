import back_view as bv
import side_view as sv
import math
import time
from flask import Flask, jsonify

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import math


app = Flask("Model")

@app.route('/<person_height>/<image_1>/<image_2>')
def size_measurement(person_height, image_1, image_2):
	person_height = float(person_height)
	
	back = bv.measurement(f"{image_1}.jpeg", person_height)
	side = sv.measurement(f"{image_2}.jpeg", person_height)
	
	chest_x = back[0]/2
	chest_y = side[0]/2

	waist_x = back[1]/2
	waist_y = side[1]/2
	
	chest = 1.5 * (back[0] + side[0])
	waist = 1.5 * (back[1] + side[1])

	print("predicted chest : ", chest/2.54)
	print("predicted waist : ", waist/2.54)

	return f"check terminal"

if __name__ == '__main__':
	app.run('127.0.0.1', 5000)

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import math
from flask import Flask, jsonify

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    if category_name != 'person':
      continue
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image, bbox.height

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    # print(pose_landmarks_proto)
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image, pose_landmarks_proto

app = Flask("Model")

@app.route('/<image_file>/<a_height>')
def measure_size(image_file, a_height):
   file_name = "DATA/" + image_file
   # Hight Measurement
   base_options_ed = python.BaseOptions(model_asset_path='efficientdet.tflite')
   options_ed = vision.ObjectDetectorOptions(base_options=base_options_ed,
                                       score_threshold=0.5)
   detector_ed = vision.ObjectDetector.create_from_options(options_ed)

   # Create an PoseLandmarker object.
   base_options_pl = python.BaseOptions(model_asset_path='pose_landmarker.task')
   options_pl = vision.PoseLandmarkerOptions(
    base_options=base_options_pl,
    output_segmentation_masks=True)
   detector_pl = vision.PoseLandmarker.create_from_options(options_pl)

   # Load the input image.
   image = mp.Image.create_from_file(file_name)

   # Detect objects in the input image.
   detection_result_ed = detector_ed.detect(image)

   # Detect pose landmarks from the input image.
   detection_result_pl = detector_pl.detect(image)

   # Process the detection result. In this case, visualize it.
   image_copy = np.copy(image.numpy_view())
   print(image.numpy_view().shape)

   annotated_image_ed, height = visualize(image_copy, detection_result_ed)
   cv2.imwrite(f"Output/{image_file.split('.')[0]}_1.jpeg", cv2.cvtColor(annotated_image_ed, cv2.COLOR_BGR2RGB))


   annotated_image_pl, pose_landmarks_proto = draw_landmarks_on_image(image.numpy_view(), detection_result_pl)
   cv2.imwrite(f"Output/{image_file.split('.')[0]}_2.jpeg", cv2.cvtColor(annotated_image_pl, cv2.COLOR_BGR2RGB))


   image_width = image.numpy_view().shape[1]
   image_height = image.numpy_view().shape[0]
   points = []
   for idx, landmark in enumerate(pose_landmarks_proto.landmark):
     print(f"Point {idx + 1}: \n")
     points.append([min(math.floor(landmark.x * image_width), image_width - 1),min(math.floor(landmark.y * image_height), image_height - 1)])
     print("x : ", min(math.floor(landmark.x * image_width), image_width - 1))
     print("y : ", min(math.floor(landmark.y * image_height), image_height - 1))
   # 11,12
   # 16,15
   img = cv2.imread(file_name)
   cv2.line(img, points[11], points[12], (0, 255, 0), 3)
   cv2.line(img, points[16], points[15], (0, 255, 0), 3)
   cv2.imwrite(f"Output/{image_file.split('.')[0]}_3.jpeg", img)

   # Size measurement
   RW1, RW2 = points[15]
   LW1, LW2 = points[16]

   dist_w = math.sqrt((LW1-RW1)**2 + (LW2-RW2)**2)
   person_height = int(float(a_height)*30.48)
   ppm = person_height/height
   print("Waist : ", dist_w*ppm, " cm")

   RS1, RS2 = points[11]
   LS1, LS2 = points[12]
   dist_s = math.sqrt((RS1-LS1)**2 + (RS2-LS2)**2)
   print("Chest : ", dist_s*ppm, " cm")  

   data = {
   "Waist" : dist_w*ppm, 
   "Chest" : dist_s*ppm 
   }

   return jsonify(data)
if __name__ == '__main__':
	app.run('127.0.0.1', 5000) 
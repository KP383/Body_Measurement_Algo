import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import math


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
    if category_name != 'person' :
      continue
    else:
      probability = round(category.score, 2)
      result_text = category_name + ' (' + str(probability) + ')'
      text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
      cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
      height = bbox.height
      x = bbox.origin_x
      y = bbox.origin_y

  return image, height, x, y

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

def measurement(image_file, a_height):
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
   # print(image.numpy_view().shape)

   annotated_image_ed, height, height_x, height_y = visualize(image_copy, detection_result_ed)
   cv2.imwrite(f"Output/{image_file.split('.')[0]}_1.jpeg", cv2.cvtColor(annotated_image_ed, cv2.COLOR_BGR2RGB))
   # print(height)

   annotated_image_pl, pose_landmarks_proto = draw_landmarks_on_image(image.numpy_view(), detection_result_pl)
   cv2.imwrite(f"Output/{image_file.split('.')[0]}_2.jpeg", cv2.cvtColor(annotated_image_pl, cv2.COLOR_BGR2RGB))

   segmentation_mask = detection_result_pl.segmentation_masks[0].numpy_view()
   visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
   cv2.imwrite(f"Output/{image_file.split('.')[0]}_4.jpeg", visualized_mask)

   image_width = image.numpy_view().shape[1]
   image_height = image.numpy_view().shape[0]
   points = []
   for idx, landmark in enumerate(pose_landmarks_proto.landmark):
     points.append([min(math.floor(landmark.x * image_width), image_width - 1),min(math.floor(landmark.y * image_height), image_height - 1)])

   start_pixel = height_y
   # print(height_x, height_y)

   for i in range(height_y-1, points[0][1]):
      data = [int(visualized_mask[i][points[8][0]][0]), int(visualized_mask[i][points[8][0]][1]), int(visualized_mask[i][points[8][0]][2])]
      if np.array_equal(data, [0,0,0]) != True :
         start_pixel_final = i+1
         break
  #  print("start_pixel_final :", start_pixel_final)
   
   for i in range(points[29][1], visualized_mask.shape[0]):
      data = [int(visualized_mask[i][points[29][0]][0]), int(visualized_mask[i][points[29][0]][1]), int(visualized_mask[i][points[29][0]][2])]
      if np.array_equal(data, [0,0,0]) != False :
         end_pixel_final_1 = i+1
         break

   for i in range(points[30][1], visualized_mask.shape[0]):
      data = [int(visualized_mask[i][points[30][0]][0]), int(visualized_mask[i][points[30][0]][1]), int(visualized_mask[i][points[30][0]][2])]
      if np.array_equal(data, [0,0,0]) != False :
         end_pixel_final_2 = i+1
         break
   
   end_pixel_final = (end_pixel_final_1 + end_pixel_final_2)/2
   person_height = (a_height*30.48)
   height_pixel = end_pixel_final - start_pixel_final 
   ppm = person_height/(height_pixel)

  #  print("end_pixel_final :", end_pixel_final)

   img = cv2.imread(file_name)
   cv2.line(img,(points[8][0], start_pixel_final), (points[8][0], int(end_pixel_final)), (0, 255, 0), 3)
  

   print("ppm (s): ", ppm)
   chest_p = int((points[11][1] + points[12][1])/2)
   waist_p = int((points[24][1] + points[23][1])/2)
   chest = chest_p + int((waist_p - chest_p)*0.20)
   waist = waist_p - int((waist_p - chest_p)*0.05)
   # print(chest)
   # print(waist)
   cv2.line(img,(0,chest), (visualized_mask.shape[1],chest), (0, 255, 0), 3)
   cv2.line(img,(0,waist), (visualized_mask.shape[1],waist), (0, 255, 0), 3)
   cv2.imwrite("side_test_1.jpeg", img)

   for i in range(0, visualized_mask.shape[1]):
      data = [int(visualized_mask[chest][i][0]), int(visualized_mask[chest][i][1]), int(visualized_mask[chest][i][2])]
      if np.array_equal(data, [0,0,0]) != True :
         start_pixel_chest = i+1
         break

   for i in range(start_pixel_chest, visualized_mask.shape[1]):
      data = [int(visualized_mask[chest][i][0]), int(visualized_mask[chest][i][1]), int(visualized_mask[chest][i][2])]
      if np.array_equal(data, [0,0,0]) != False :
         end_pixel_chest = i+1
         break

   for i in range(0, visualized_mask.shape[1]):
      data = [int(visualized_mask[waist][i][0]), int(visualized_mask[waist][i][1]), int(visualized_mask[waist][i][2])]
      if np.array_equal(data, [0,0,0]) != True :
         start_pixel_waist = i+1
         break
   

   for i in range(start_pixel_waist, visualized_mask.shape[1]):
      data = [int(visualized_mask[waist][i][0]), int(visualized_mask[waist][i][1]), int(visualized_mask[waist][i][2])]
      if np.array_equal(data, [0,0,0]) != False :
         end_pixel_waist = i+1
         break

   chest_length = end_pixel_chest - start_pixel_chest
   waist_length = end_pixel_waist - start_pixel_waist
   
   start_pixel_chest = int(start_pixel_chest + chest_length*0.025)
   end_pixel_chest = int(end_pixel_chest - chest_length*0.025)

   start_pixel_waist = int(start_pixel_waist + waist_length*0.025)
   end_pixel_waist = int(end_pixel_waist - waist_length*0.025)

   cv2.line(visualized_mask,(start_pixel_chest, chest), (end_pixel_chest, chest), (0, 0, 255), 3)
   cv2.line(visualized_mask,(start_pixel_waist, waist), (end_pixel_waist, waist), (0, 0, 255), 3)

   cv2.imwrite("side_test_2.jpeg", visualized_mask)
   
   chest_length = (end_pixel_chest - start_pixel_chest)
   waist_length = (end_pixel_waist - start_pixel_waist)

   print("chest_length (s):",chest_length*ppm)
   print("waist_length (s):", waist_length*ppm)

   return (chest_length*ppm, waist_length*ppm)
# measurement("76.jpeg", 5.1)
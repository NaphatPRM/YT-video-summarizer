## Import all library
## Part 1 : Scene and Image Segmentation
import streamlit as st
import numpy as np
import tempfile
import cv2
cv2.Mat = np.typing.NDArray[np.uint8]
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

## Part 1.5 : ChatGPT Baseline Testing
import base64
import requests
from transformers import pipeline

## Part 2 : Object and Character Detection
from ultralytics import YOLO
from PIL import Image as im
import os
HOME = os.getcwd()

## Part 3 : Emotion Analysis
from deepface import DeepFace

## Part 4 : Background Analysis
from transformers import pipeline

## Part 5 : Camera Analysis
### 5.1
from numpy import linalg as LA
### 5.2.1
import torch
# import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#import torchvision
# from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
## 5.2.2
import av
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
# from huggingface_hub import hf_hub_download
from scipy.spatial.distance import cdist
# from torchvision.models.video import r3d_18
### 5.3
import imutils
#from imutils import paths

### Part 6 : Dialogue Summarization
# import moviepy.editor as mp
import speech_recognition as sr

## Testing the Video Spitting
def splitting_video(video_path, detector):
  """
    Function used to split the video into a snippet based on the cut.
    Args:
        video_path (string): The file path to the image in the local
        detector (Detection algorithms): The algorithm used to detect the scene
    Returns:
        list of string: List of splitted videos
  """
  ## Perform the video split
  scene_list = detect(video_path, detector)
  split_video_ffmpeg(video_path, scene_list)

  ## Getting the name list of the videos to use later in path finding
  list_name_video = []

  for i in range(len(scene_list)):
    ## Since the notation is 3 digits (such as 001, 012)
    number = str(i+1)
    while len(number) < 3:
      number = "0" + number

    ## Add the video list name
    list_name_video.append(video_path.split("/")[1].split(".")[0] + "-Scene-" + number + ".mp4")
  return list_name_video
# splitting_video("/content/drive/MyDrive/CIS_5810/Final_Project/dolly_zoom.mp4", AdaptiveDetector())

def save_uploaded_file(image, filename):
    """
    Save the given image as a file.
    Args:
        image (numpy array): The frame to save
        filename (string): The desired file name
    Returns:
        filepath (string): Path to the saved file
    """
    filepath = os.path.join(tempfile.gettempdir(), filename)
    cv2.imwrite(filepath, image)
    return filepath
    
# Getting the first frame and check the image
# If we want to improve to the sequence of image, just return as the
def get_needed_frame(video_target):
    """
    Get the two frames to perform further contents in the videos.
    Args:
        video_target (string): The file path to the video in the local system
    Returns:
        first_frame (string): The first frame we need from the video
        last_frame (string): The last frame we need from the video
    """
    # Use VideoCapture to get frames
    vidcap = cv2.VideoCapture(video_target)
    marker = video_target.split('/')[-1].split('.')[0]
    success, image = vidcap.read()
    count = 0

    prev_image = None
    first_frame = None

    # Reading frames until the end
    while success:
        if count == 2:
            first_frame = save_uploaded_file(image, f"first_frame_{marker}_{count}.jpg")
        if count == 30:
            break
        success, image = vidcap.read()
        count += 1
        if success:
            prev_image = image

    # Save the last frame
    last_frame = save_uploaded_file(prev_image, f"last_frame_{marker}_{count}.jpg") if prev_image is not None else None
    
    return first_frame, last_frame

## Snapshot Summarization API
## Plan : Capture the start point of the video to grab on the settings
def encode_image(image_path):
  """
    Encoding the image from the local that it could work with the ChatGPT API.
    Args:
        image_path (string): The file path to the image in the local

    Returns:
        base 64 string: The base64 string of the image
    """
  with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

def human_description(image_path, api_key):
  """
  The function used to extract the contents of the human in the video (which we consider only the first frame)
  Args:
      image_path (string): The file path to the image in the local

  Returns:
      string describing the human appearance in the video, return as there is no human if there is none.
  """

  # Getting the base64 string
  base64_image = encode_image(image_path)

  # Calling the ChatGPT API
  payload =  {"model": 'gpt-4o',
    "messages": [
    {"role": "system",
      "content": [{"type": "text",
                  "text": "You are an image analyst. Your goal is to describe the human in this image."}],
    },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe all the human (and only human) in the image. If there is no human in the image, please simply say there is no human."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 500
  }


  # Headers : Determine the type of contents return and the key
  headers = {"Authorization": f"Bearer " + api_key,
            "Content-Type": "application/json"}

  # Getting the response back from the ChatGPT API
  response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
  r = response.json()

  return r["choices"][0]["message"]["content"]


## Second Approach : How about picking the HuggingFace instead?
def overall_analysis(image_check):
  captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
  dict_result = captioner(image_check)
  return dict_result[0]["generated_text"]

# Object Semantic Segmentation and detecting the face
# Define a function of saving the picture of image
def save_face_image(img_matrix, label_face, suffix):

  # Getting the cropped index
  object_return = [int(label_face[0]), int(label_face[1]), int(label_face[2]), int(label_face[3])]
  x,y,w,h = object_return

  # Converting from the cropped numpy to image
  face_1 = im.fromarray(img_matrix[y:y+h, x:x+w])

  # Saving the image
  name_image = "face" + str(suffix) + ".jpg"
  face_1.save(name_image)
  return name_image

def object_detection(first_frame):
  # Load a model
  model = YOLO(f'{HOME}/yolov8n.pt')

  # Perform object detection on an image
  results = model.predict(source=first_frame, conf=0.5)

  # Converting the image to the numpy array
  img = cv2.imread(first_frame)

  # Running through to get the object
  # Initialize the data
  list_name = []
  dict_objects = {}
  object_string = ""

  # Setting the var
  dict_name = results[0].names
  num_face = 1

  # Running the loop for the contents in the image
  for i in range(len(results[0].boxes.cls)):
    if results[0].boxes.cls[i] == 0:
      new_name = save_face_image(img, results[0].boxes.xyxy[i], num_face)
      num_face += 1
      list_name.append(new_name)
    else:
      get = dict_name[int(results[0].boxes.cls[i])]
      if get in dict_objects:
        dict_objects[get] += 1
      else:
        dict_objects[get] = 1

  # In case if there is an info in the dict_contect, we'll add it into the object_string
  if len(dict_objects) > 0:
    object_string = "This image includes "
    for key, value in dict_objects.items():
      object_string += str(value) + " " + key + ", "
    object_string = object_string[:-2] + "."

  return list_name, object_string

# Face Emotion detection
# In this case, the emotion detected would be Happiness, Surprise, Anger, Sadness, Disgust, Fear, Neutral
def ordinal_cardinal(i):
  if i in [11, 12, 13]:
    return str(i) + "th"
  list_ending = ["th", "st", "nd", "rd", "th"]
  return str(i) + list_ending[min(i%10, 4)]

def emotion_description(image_used, i=0):
  # may need to do the threshold for the probability
  objs = DeepFace.analyze(
    img_path = image_used,
    actions = ['emotion'],
    enforce_detection = False
  )

  dict_human = objs[0]
  return "The " + ordinal_cardinal(i+1) + " person feels " + dict_human["dominant_emotion"] + "."

## New Idea : Using the Homography Analysis from the first and last frame instead to determine the change of the camera
## Block 1 : Function for the Homography

def compute_homography(p1, p2):
    """
    Compute homography transform matrix given 4 pairs of corresponding points

    Input:
      p1, p2: (4, 2) shape of 4 corresponding points

    Output:
      H: (3, 3) shape of homography matrix such that lambda * p2 = H @ p1
    """
    # Step 1: Initialize the matrix A
    matrix_A = np.zeros((8, 9))

    # Step 2: Populate the matrix A with the correct point correspondences
    for i in range(4):
        x1, y1 = p1[i, :]
        x2, y2 = p2[i, :]
        matrix_A[2 * i] = [x2, y2, 1, 0, 0, 0, -x2 * x1, -y2 * x1, -x1]
        matrix_A[2 * i + 1] = [0, 0, 0, x2, y2, 1, -x2 * y1, -y2 * y1, -y1]

    # Step 3: Perform SVD on matrix A to solve Ah = 0
    U, S, Vt = np.linalg.svd(matrix_A)
    h = Vt[-1]  # The last row of Vt corresponds to the smallest singular value

    # Step 4: Reshape h into the homography matrix H
    H = h.reshape((3, 3))

    # Step 5: Normalize so that H[2,2] = 1
    H /= H[2, 2]

    return H

def match_features(f1, f2):
    """
    Match two sets of features

    Input:
    f1, f2: (N, feature_size) shape of features to be matched

    Output:
    match, match_fwd, match_bkwd: (N, 2) shape of final matching result, forward matching result and backward matching result
    For each matching result, the first column is the index in f1 and the second column is the index in f2
    """
    # Step 1: Compute pairwise distance between f1 and f2
    distance = cdist(f1, f2)

    # Step 2: Perform forward matching
    # Get two closest matches for each feature in f1 (sorted indices)
    forward_matches = np.argsort(distance, axis=1)

    # Get the closest two distances and perform the ratio test
    fwd_ratio = np.take_along_axis(distance, forward_matches, axis=1)[:, 0] / np.take_along_axis(distance, forward_matches, axis=1)[:, 1]

    # Apply ratio test (using a threshold of 0.8 for example)
    match_fwd = np.array([[i, forward_matches[i, 0]] for i in range(distance.shape[0]) if fwd_ratio[i] < 0.7])

    # Step 3: Perform backward matching
    # Get two closest matches for each feature in f2
    backward_matches = np.argsort(distance, axis=0)

    # Get the closest two distances and perform the ratio test
    bwd_ratio = np.take_along_axis(distance, backward_matches, axis=0)[0] / np.take_along_axis(distance, backward_matches, axis=0)[1]

    # Apply ratio test
    match_bkwd = np.array([[backward_matches[0, i], i] for i in range(distance.shape[1]) if bwd_ratio[i] < 0.7])

    # Step 4: Find intersection of forward and backward matches for final matching result
    match = np.array([m for m in match_fwd if [m[0], m[1]] in match_bkwd.tolist()])

    return match, match_fwd, match_bkwd

def ransac_homography(p1, p2):
  """
  Estimate homography matrix with RANSAC

  Input:
  p1, p2: (N, 2) shape of correponding points

  Output:
  H: (3, 3) shape of estimated homography matrix such that lambda * p1 = H @ p2
  """
  # IMPLEMENT HERE
  # Decide how many loops to run and what the threshold is

  p = 1 - 1e-5 # Success rate with the threshold = 0.0001
  w = 0.5 # Fine tuning the ratio of the inlier and the samples
  num_iterations = int(np.log(1 - p) / np.log(1 - w ** 4))
  best_H = None
  best_inliers = 0

  # Convert p1 and p2 to homogeneous coordinates
  p2_full = np.hstack((p2, np.ones((p2.shape[0], 1))))

  # RANSAC loop, inside in the loop
  for i in range(num_iterations):
    # 1) Randomly pick n samples from p1 and p2 that is enough to fit a model (n=4 here)

    select_list = np.random.choice(p1.shape[0], 4, replace=False)
    p1_ = p1[select_list]
    p2_ = p2[select_list]

    # 2) Fit the model and get a estimation
    H = compute_homography(p1_, p2_)

    # 3) Apply the homography to all points in p1
    projected_p1 = (H @ p2_full.T).T  # Homography transformation
    projected_p1 = projected_p1[:, :2] / projected_p1[:, -1, None]  # Convert back to Cartesian coordinates

    # 4) Compute reprojection error (Euclidean distance between p2 and projected p1)
    match, match_fwd, match_bkwd = match_features(p1, projected_p1)

    # 5) Identify inliers where the reprojection error is less than the threshold
    num_inliers = match.shape[0]

    # 4) Update the best estimation if the current one is better
    if num_inliers > best_inliers:
      best_H = H
      best_inliers = num_inliers

  return best_H


def matrix_analysis(first_frame, last_frame):
  # load images in OpenCV BGR format
  I1 = cv2.imread(first_frame)
  I2 = cv2.imread(last_frame)

  # create grayscale images
  I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
  I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY);

  # convert images to RGB format for display
  I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
  I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)

  # compute SIFT features
  sift = cv2.SIFT_create()
  kp1, des1 = sift.detectAndCompute(I1_gray, None)
  kp2, des2 = sift.detectAndCompute(I2_gray, None)

  # match features
  match, match_fwd, match_bkwd = match_features(des1, des2)

  # get corresponding points p1, p2
  p1 = np.array([kp.pt for kp in kp1])[match[:, 0]]
  p2 = np.array([kp.pt for kp in kp2])[match[:, 1]]

  # estimate homography transform with RANSAC
  try:
    H = ransac_homography(p1, p2)
    K, _ = cv2.findFundamentalMat(p1, p2)

    # Finding the eigenvalues and eigenvector
    eigenvalues, eigenvectors = LA.eig(H)
    return eigenvalues, eigenvectors, H, K
  except:
    return None, None, None, None

def camera_movement(first_frame, last_frame):
    eigenvalues, eigenvectors, H, K = matrix_analysis(first_frame, last_frame)
    num_frames = int(last_frame.split(".")[0].split("_")[-1]) - int(first_frame.split(".")[0].split("_")[-1])

    if H is None:
        return f"whip pan in {num_frames} frames"

    # Check for complex eigenvalues to determine Pan or Tilt
    if np.any(np.iscomplex(eigenvalues)):
        # Find the index of the real eigenvalue
        real_idx = np.where(~np.iscomplex(eigenvalues))[0]
        eigenvector = eigenvectors[:, real_idx].flatten()
        angle = round((np.angle(eigenvalues[0] / eigenvalues[2]) / np.pi) * 180, 2)

        if eigenvector[0] > eigenvector[1]:
            return f"panning with an angle of {angle} degrees"
        else:
            return f"tilting with an angle of {angle} degrees"

    # Check transformation characteristics in H
    if abs(H[0, 0] - H[1, 1]) >= 0.5:
        return f"whip pan in {num_frames} frames"

    if abs(H[0, 0]**2 + H[0, 1]**2 - 1) <= 0.05:
        if abs(H[0, 0] - 1) <= 0.02:
            return "static"
        else:
            return f"camera roll in {num_frames} frames"

    # Handle Zoom
    scale = (H[0, 0] + H[1, 1]) / 2
    offset_x = H[0, 2] / (1 - scale)
    offset_y = H[1, 2] / (1 - scale)
    if scale > 1.0:
        return f"zoom in with a scale of {scale:.2f} centered on ({offset_x:.2f}, {offset_y:.2f})"
    else:
        return f"zoom out with a scale of {scale:.2f} centered on ({offset_x:.2f}, {offset_y:.2f})"

# Predicting the camera movement from the videos
# Using the testing examples from the hugging face
np.random.seed(0)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = hf_hub_download(
#     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
# )
def camera_action(file_path):
  try:
    container = av.open(file_path)

    # sample 16 frames
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = list(read_video_pyav(container, indices))

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

    inputs = processor(list(video), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # model predicts one of the 400 Kinetics-400 classes
    predicted_label = logits.argmax(-1).item()
    action = model.config.id2label[predicted_label]
    return "In this video, the action is " + action
  except:
    return "No action"


# import the necessary packages
def find_marker(image_name):
	image = cv2.imread(image_name)
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def camera_focal_length(first_frame):
  # initialize the known distance from the camera to the object, which
  # in this case is 24 inches
  KNOWN_DISTANCE = 24.0
  # initialize the known object width, which in this case, the piece of
  # paper is 8 inches wide
  KNOWN_WIDTH = 8.0
  # load the furst image that contains an object that is KNOWN TO BE 2 feet
  # from our camera, then find the paper marker in the image, and initialize
  # the focal length
  marker = find_marker(first_frame)
  focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
  return focalLength

## Flair? : Adding Video Speech to Text + Timestamp?
## Use Off-the-shelf due to the part is unrelated to the Computer Vision

# Load the video
def transcript_video(full_video):
  video = mp.VideoFileClip(full_video)

  # Extract the audio from the video
  audio_file = video.audio
  audio_file.write_audiofile("my_video.wav")

  # Initialize recognizer
  r = sr.Recognizer()

  # Load the audio file
  with sr.AudioFile("my_video.wav") as source:
      data = r.record(source)

  # Convert speech to text
  text = r.recognize_google(data)

  # Print the text
  print("\nThe resultant text from video is: \n")
  print(text)

# Summarizing the description of the videos
# All results

def final_summary(check, api_key):
  payload =  {"model": 'gpt-4o',
      "messages": [
      {"role": "system",
        "content": [{"type": "text",
                    "text": "You are a text summarizer. Your goal is based on the set of sentences or paragraphs I give you, return the summary that capture the description I give the most"}],
      },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "From the given text, please summarize the text as short as possible while remain the content"
            },
            {
              "type": "text",
              "text": check
            }
          ]
        }
      ],
      "max_tokens": 500
    }



  headers = {"Authorization": f"Bearer " + api_key,
              "Content-Type": "application/json"}


  response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
  r = response.json()
  return r["choices"][0]["message"]["content"]

## Function for the one snippet
def one_snippet_check(video_target, api_key):
  ## 1. Get the frame we want
  first_frame, last_frame = get_needed_frame(video_target)
  ## 2. Object and Character Detection
  result_faces, object_description = object_detection(first_frame)
  ## 4. Background analysis
  human_des = human_description(first_frame, api_key)
  ## 5. Emotional Analysis
  result_face = ""
  for i in range(len(result_faces)):
    result_face += emotion_description(result_faces[i], i)
  ## 6. Camera Analysis
  camera_move_description = camera_movement(first_frame, last_frame)
  camera_action_description = camera_action(video_target)
  camera_focal_length_description = camera_focal_length(first_frame)
  ## 7. Summarizing
  total_gen_description = object_description + " " + human_des + " " + result_face + " " + camera_action_description
  return (first_frame, final_summary(total_gen_description, api_key), camera_move_description, camera_focal_length_description)

def video_summarization(full_video, api_key, limit=None):
  st.success("Processing local video file....")
  scene_list = splitting_video(full_video, AdaptiveDetector())
  list_total_result = []
  if limit != None:
    scene_list = scene_list[:limit+1]
  for video in scene_list:
    list_total_result.append(one_snippet_check(video, api_key))
  return list_total_result

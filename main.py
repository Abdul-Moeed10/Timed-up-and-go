import numpy as np
import cv2
import tensorflow as tf
from utils.configurations import KEYPOINT_DICT as kp_names
from utils.predictor import preprocess, get_prediction

def build_interpreter(path):
    model = tf.keras.models.load_model(path)
    return model.signatures["serving_default"]

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def is_standing(left_knee_angle, right_knee_angle):
    return left_knee_angle > 160 or right_knee_angle > 160

def is_sitting(left_knee_angle, right_knee_angle):
    return left_knee_angle < 130 or right_knee_angle < 130

videofile = input('Enter video path: ')

interpreter = build_interpreter(path='model')
cap = cv2.VideoCapture(videofile)

person_got_up = False
start_time = None
end_time = None

subject_standing = False
subject_sitting = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor, frame, _ = preprocess(frame[...,::-1], input_size=(512, 512))
    fw, fh, _ = frame.shape
    kpts, _, _, _, _ = get_prediction(input_tensor, interpreter, from_class=1)
    kpts = kpts[...,::-1] * np.array([fw, fh])

    if kpts.shape[0] == 0:
        continue

    left_hip = kpts[0, kp_names['left_hip']]
    left_knee = kpts[0, kp_names['left_knee']]
    left_ankle = kpts[0, kp_names['left_ankle']]

    right_hip = kpts[0, kp_names['right_hip']]
    right_knee = kpts[0, kp_names['right_knee']]
    right_ankle = kpts[0, kp_names['right_ankle']]

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


    if 5 <= current_time <= 7:
        if not person_got_up:
            if is_standing(left_knee_angle, right_knee_angle):
                person_got_up = True
                subject_standing = True
                start_time = current_time
                print(f"Person has gotten up at {start_time} seconds")

    if 13 <= current_time <= 16:
        if person_got_up:
            if is_sitting(left_knee_angle, right_knee_angle):
                end_time = current_time
                subject_sitting = True
                print(f"Person has sat down at {end_time} seconds")
                break

    for kp in kpts[0]:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if start_time is not None and end_time is not None:
    total_time = end_time - start_time
    if subject_standing and subject_sitting:
        print("Total time from getting up to sitting down:", total_time, "seconds")
    else:
        print("The person did not complete the activity.")

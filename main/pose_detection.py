import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd
import os.path
import pickle

# load model
with open('./saved_model/body_language.pkl', 'rb') as f:
    model = pickle.load(f)

def pose_detection(video_path=0,train=True, class_name="training_classifier_name"):
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    pose_name_fps = []
    pose_prob_fps = []

    cap = cv2.VideoCapture(video_path)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        c = 0
        while cap.isOpened():
            
            ret, frame = cap.read()
            
            if ret == True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                        
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 1. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Concate rows
                    row = pose_row
                    
                    if train==True:

                        # Create file if not present
                        if os.path.isfile(r'./data/coords.csv'):
                            pass
                        # Capture Landmarks & Export to CSV
                        else:
                            num_coords = len(results.pose_landmarks.landmark)

                            landmarks = ['class']
                            for val in range(1, num_coords+1):
                                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

                            with open(r'./data/coords.csv', mode='w', newline='') as f:
                                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(landmarks)

                        # Append class name 
                        row.insert(0, class_name)

                        # Export to CSV   
                        with open(r'./data/coords.csv', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row) 
                    
                    # if train is not true do the prediction
                    else:
                        # Make Detections
                        X = pd.DataFrame([row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        #print(body_language_class, body_language_prob)
                        pose_name_fps.append(body_language_class)
                        pose_prob_fps.append(body_language_prob)

                        # Grab ear coords
                        coords = tuple(np.multiply(
                                        np.array(
                                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                    , [640,480]).astype(int))
                        
                        cv2.rectangle(image, 
                                    (coords[0], coords[1]+5), 
                                    (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                    (245, 117, 16), -1)
                        cv2.putText(image, body_language_class, coords, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Get status box
                        cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                        
                        # Display Class
                        cv2.putText(image, 'CLASS'
                                    , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, body_language_class.split(' ')[0]
                                    , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Display Probability
                        cv2.putText(image, 'PROB'
                                    , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                    , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                except:
                    pass
                                
                cv2.imshow('Video', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    pose_prob_fps = np.array(pose_prob_fps)
    cap.release()
    cv2.destroyAllWindows()
    return (pose_name_fps, pose_prob_fps, model.classes_)
from flask import Flask,render_template,url_for,request
import pandas as pd 
from werkzeug.utils import secure_filename
import timeit

import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd
import os.path
import pickle

from main import pose_detection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'./upload/'

def moving_average(a, n=40):
    """
    Computes the moving average of our probabilities over 'n' rows.
    Not actually doing the efficient moving average for faster coding, slower execution which will be negligible.
    """
    averages = []
    for i in range(len(a)-n+1):
        averages.append(a[i:i+n].mean(axis=0))
    return np.array(averages)

def pose_selection(pose_prob, poses):
    selected_poses = []
    for row in pose_prob:
        selected_poses.append({'pose': poses[np.argmax(row)], 'probability': max(row)})
    return selected_poses

def count_reps(poses, detected_poses, threshold=0.25):
    count = {}
    for pose in poses:
        count[pose] = 0
    previous = ''
    for frame in detected_poses:
        pose = frame['pose']
        if frame['probability'] < threshold:
            continue
        if pose == previous:
            continue
        count[pose] += 1
        previous = pose
    return count

def count_time(poses, detected_poses, fps=24, threshold=0.25):
    count = {}
    total = 0
    for pose in poses:
        count[pose] = 0
    for frame in detected_poses:
        pose = frame['pose']
        if frame['probability'] > threshold:
            count[pose] += 1/fps
            total += 1
    for frame in detected_poses:
        count[frame['pose']] = np.round(count[frame['pose']])
    print(total)
    return count

@app.route('/', methods=['GET', 'POST'])
def dash_board():
    if request.method == 'GET':
        return render_template("index.html")
    
    elif request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        age = request.form['age']
        weight = request.form['weight']
        height = request.form['height']
        feed_type = request.form['feed_type']

        if feed_type=='video_file':
            file = request.files['file_upload']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(feed_type)
            # Prediction
            pose_history = pose_detection.pose_detection(video_path=os.path.join(app.config['UPLOAD_FOLDER'], filename),
            train=False)

            pose_name_fps, pose_prob_fps, model_class = pose_history
            pose_prob_fps_averages = moving_average(pose_prob_fps)

            selected_p = pose_selection(pose_prob_fps, model_class)
            selected_p_averages = pose_selection(pose_prob_fps_averages, model_class)

            total_reps = count_reps(model_class, selected_p_averages)
            total_reps = sum(total_reps.values())
            
            total_time = count_time(model_class, selected_p_averages)
            total_time = sum(total_time.values())

        else:
            print('Live Feed')
            start = timeit.default_timer()
            pose_history = pose_detection.pose_detection(video_path=0,
            train=False)
            stop = timeit.default_timer()

            total_reps = 0
            
            total_time = stop - start
            total_time = round(total_time/60,1)
        
        return render_template('Dashboard.html',f_name=first_name,l_name=last_name,age_info=age,weight_info=weight,height_info=height, total_pose=len(set(pose_history[0])), total_time=total_time, total_reps=total_reps)


if __name__ == '__main__':
	app.run(debug=True)
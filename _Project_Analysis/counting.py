# Prediction
pose_history = pose_detection(video_path=training_vid_path,
train=False)
# Deploy

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
pose_name_fps, pose_prob_fps = pose_history

pose_prob_fps_averages = moving_average(pose_prob_fps)

'''plt.figure(figsize=(12, 8))

plt.plot(pose_prob_fps)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(pose_prob_fps_averages[35000:40000])
plt.legend(model.classes_, )
plt.show()'''

count_reps(model.classes_, selected_p_averages)
count_time(model.classes_, selected_p_averages)

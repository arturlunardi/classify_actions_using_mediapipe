import cv2
import numpy as np
import pandas as pd
import mediapipe as mp 
import pickle
import os
import utils


def rescale_frame(frame, scale):    # works for image, video, live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def display_classify_pose(cap, model):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
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

                # Pose Detections
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

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(f'class: {body_language_class}, prob: {body_language_prob}')

                    # Grab ear coords
                    coords = tuple(np.multiply(
                        np.array(
                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                        [640,480]
                    ).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(
                        image, 'CLASS', (95,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, body_language_class.split(' ')[0], (90,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    # Display Probability
                    cv2.putText(
                        image, 'PROB', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), 
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                except:
                    pass

                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            else:
                break

    print('Done!')
    cap.release()
    cv2.destroyAllWindows()


def save_display_classify_pose(cap, model, out_video, scale=0.4):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = input_fps - 1
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'video_w: {w}, video_h: {h}')

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 輸出附檔名為 mp4. 
    # out = cv2.VideoWriter(out_video, fourcc, output_fps, (w, h))


    out = cv2.VideoWriter(out_video, 0x00000021, output_fps, (w, h))
    
    
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame = rescale_frame(frame, scale)
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False    

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # # 1. Draw face landmarks
                # mp_drawing.draw_landmarks(
                #     image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                #     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                #     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                # )
                # # 2. Right hand
                # mp_drawing.draw_landmarks(
                #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                #     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                #     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                # )
                # # 3. Left Hand
                # mp_drawing.draw_landmarks(
                #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                #     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                #     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                # )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # # Extract Face landmarks
                    # face = results.face_landmarks.landmark
                    # face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                    # # Extract Right Hand landmarks
                    # r_hand = results.right_hand_landmarks.landmark
                    # r_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in r_hand]).flatten())   

                    # # Extract left Hand landmarks
                    # l_hand = results.left_hand_landmarks.landmark
                    # l_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in l_hand]).flatten())                                        
                    
                    # Concate rows
                    # row = pose_row+face_row+r_hand_row+l_hand_row
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(f'class: {body_language_class}, prob: {body_language_prob}')

                    # Grab ear coords
                    coords = tuple(np.multiply(
                        np.array(
                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                        [640,480]
                    ).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    # cv2.rectangle(image, (0,0), (300, 60), (245, 117, 16), -1)
                    cv2.rectangle(image, (0,0), (len('CLASS')*20+len(body_language_class)*20, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(
                        image, 'CLASS', (145,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, body_language_class.split(' ')[0], (130,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    # Display Probability
                    cv2.putText(
                        image, 'PROB', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA
                    )
                    body_language_prob = body_language_prob*100
                    cv2.putText(
                        image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), 
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                except:
                    pass

                out.write(image)
                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            else:
                break

    print('Done!')
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    videos_train_path = os.path.abspath(os.path.join(__file__, utils._root_data_path, utils._video_training_path))
    output_videos_path = os.path.abspath(os.path.join(__file__, utils._root_data_path, utils._output_display_model_videos_path))
    model_weights = os.path.abspath(os.path.join(__file__, utils._root_model_weight_path, utils._model_weight_file))

    video_file_name = "putting_on_shoes"

    video_path = os.path.join(videos_train_path, video_file_name + ".mp4")
    output_video = os.path.join(output_videos_path, video_file_name + "_out.mp4")

    cap = cv2.VideoCapture(video_path)

    # Load Model.
    with open(model_weights, 'rb') as f:
        model = pickle.load(f)
    
    # display_classify_pose(cap=cap, model=model)
    save_display_classify_pose(cap=cap, model=model, out_video=output_video, scale=1)
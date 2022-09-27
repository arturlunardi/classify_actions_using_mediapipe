# About

This repository use [MediaPipe](https://google.github.io/mediapipe/) to build a model that classifies actions based on human activities.

![display_model_out](https://media.giphy.com/media/7nReEIXBvpmPQuYIIA/giphy.gif)

# Methodology

## Data

The data was downloaded from the [Kinetics Dataset](https://www.deepmind.com/open-source/kinetics). The dataset contains labeled human action classes with URL links to download. Each clip is human annotated with a single action class and lasts around 10 seconds.

In this project, we trimmed the dataset into 20 classes and selected the first video of each class. Then, we used ffmpeg to download the video from the url's.

## Pipeline

To process the data and extract the features, we used the [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) framework.

## Model

To create our features, we extract the [pose landmarks](https://google.github.io/mediapipe/solutions/pose.html) for every frame. Holistic also allows you to extract Face and Hands landmarks.

For every video used in training, we extract the pose landmarks for every frame and create a dataset to predict the actions based on the coordinates. You can also train your own videos or use a webcam to do it.

The model is trained in a few algorithms, but the chosen one was RandomForest Classifier. This model will later be used to predict an action on every frame of a video or a webcam.

# Tree Structure

<pre>
├───application
│   │   create_dataset_actions.py
│   │   display_model.py
│   │   download_videos.py
│   │   model_train.py
│   └───utils.py
│
├───data
│   │   coords_dataset.csv
│   │   train.csv
│   │
│   ├───output_display_model_videos
│   │       picking_apples_out.mp4
│   │       putting_on_shoes_out.mp4
│   │
│   └───video_trainings
│           adjusting_glasses.mp4
│           brushing_floor.mp4
│           capsizing.mp4
│           chasing.mp4
│           closing_door.mp4
│           combing_hair.mp4
│           cutting_cake.mp4
│           dealing_cards.mp4
│           grooming_cat.mp4
│           metal_detecting.mp4
│           peeling_banana.mp4
│           petting_horse.mp4
│           picking_apples.mp4
│           polishing_furniture.mp4
│           putting_on_shoes.mp4
│           saluting.mp4
│           sucking_lolly.mp4
│           taking_photo.mp4
│           tasting_wine.mp4
│
└───model_weights
        weights_body_language.pkl
</pre>

## File Description

- **download_videos.py** - Download the first video of every selected classes on the Kinetic Dataset.

- **create_dataset_actions.py** - Check if the csv dataset is exist or not. If not, create it. If exists, add the pose landmarks based on the video file name (label of the class).

- **model_train.py** - Train the model based on the coordinates dataset.

- **display_model.py** - Predict the actions on every frame of the video and output the predicted video to the output_display_model_videos directory.

- **utils.py** - path of files and directories.

# Usage

1. Clone the Repository

2. Access the project directory and install the requirements.

```
pip install -r requirements.txt
```

3. If you want to download videos, download [ffmpeg](https://ffmpeg.org/), add it to the PATH and copy the ffmpeg.exe to the application directory. 

4. To predict the actions on a video, open the display_model file, set the video_file_name to predict, go to the file directory and run

```
python display_model.py
```

You can also predict using a webcam, setting the `cv2.VideoCapture(arg)` arg to 0.

## Rebuild

If you want to select other classes, increase the number of files to be download or just retrain the model, you can edit the `download_videos` and the `create_dataset_actions` files. 

Then, in order, run the `download_videos` > `create_dataset_actions` > `model_train` > `display_model`.

In this project we are considering that every file name it is a class to create the dataset. But we could download a massive number of videos with a unique name and move it to a directory for every class to create the dataset.

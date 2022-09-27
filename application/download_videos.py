import pandas as pd
import youtube_dl
import subprocess
import os
import datetime
import utils


def main():
    """
    The function will run through every record of the df_to_download DataFrame and use ffmpeg to download the video from the url.
    """    
    # load original data
    df_videos = pd.read_csv(os.path.abspath(os.path.join(__file__, utils._root_data_path, utils._train_video_file)))

    # defining desired labels
    chosen_labels = [
        'cutting cake',
        'saluting',
        'adjusting glasses',
        'chasing',
        'grooming cat',
        'polishing furniture',
        'tasting wine',
        'putting on shoes',
        'peeling banana',
        'looking at phone',
        'taking photo',
        'combing hair',
        'brushing floor',
        'dealing cards',
        'closing door',
        'picking apples',
        'capsizing',
        'sucking lolly',
        'petting horse',
        'metal detecting'
    ]

    df_to_download = pd.DataFrame()
    # accessing each label and geting only the first record to make train faster
    for label in chosen_labels:
        first_record = df_videos.loc[df_videos['label'] == label].iloc[0]
        df_to_download = df_to_download.append([first_record])

    # acessing each row to download the video
    for index, row in df_to_download.iterrows():
        URL = f"https://www.youtube.com/watch?v={row['youtube_id']}"
        start_time = datetime.timedelta(seconds=row['time_start'])
        end_time = datetime.timedelta(seconds=row['time_end'])
        TARGET_LABEL = f"{'_'.join(row['label'].split())}.mp4"
        TARGET_PATH = os.path.abspath(os.path.join(__file__, utils._root_data_path, utils._video_training_path, TARGET_LABEL))

        with youtube_dl.YoutubeDL({'format': 'best'}) as ydl:
            result = ydl.extract_info(URL, download=False)
            video = result['entries'][0] if 'entries' in result else result

        raw_url = video['url']
        # using ffmpeg to download videos
        subprocess.call('ffmpeg -i "%s" -ss %s -t %s -c:v copy -c:a copy "%s"' % (raw_url, str(start_time.seconds), str((end_time - start_time).seconds), TARGET_PATH))


if __name__ == '__main__':
    main()
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import cv2

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):

        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        #  Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        self.emotion_map = {
            'anger' : 0,
            'disgust': 1,
            'sadness' : 2,
            'joy' : 3,
            'neutral': 4,
            'surprise' : 5,
            'fear' : 6
        }

        self.sentiment_map = {
            'negative' : 0,
            'neutral' : 1,
            'positive' : 2
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
       row = self.data.iloc[index]
       video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
       
       path = os.path.join(self.video_dir, video_filename)
       video_path = os.path.exists(path)

       if video_path == False:
           raise FileNotFoundError(f"No video found for: {path}")
        
       text_input = self.tokenizer(row['Utterance'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')

       print(text_input)

    def _load_video_frames(self, video_path):
        return ""


if __name__ == "__main__":

    meld = MELDDataset(
        '../dataset/dev/dev_sent_emo.csv',
        '../dataset/dev/dev_splits_complete'
    )
    print(meld[4])


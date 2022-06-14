import os
import pandas as pd
import numpy as np
import datetime
from tensorflow import keras
from keras.utils import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input, VGG16


CWD = os.getcwd()

def train_test_split(faces_vgg: str, train: str, cv: str, test: str):
    """Create train, test and validation split"""
    faces_vgg_df = pd.read_parquet(CWD + faces_vgg)
    perm_idx = np.random.permutation(faces_vgg_df['name'].unique())
    train_ind = perm_idx[:int(len(perm_idx)*0.6)]
    cv_ind = perm_idx[int(len(perm_idx)*0.6):int(len(perm_idx)*0.8)]
    test_ind = perm_idx[int(len(perm_idx)*0.8):]
    faces_vgg_df[faces_vgg_df['name'].isin(train_ind)].to_parquet(CWD + train, index=False)
    faces_vgg_df[faces_vgg_df['name'].isin(cv_ind)].to_parquet(CWD +cv, index=False)
    faces_vgg_df[faces_vgg_df['name'].isin(test_ind)].to_parquet(CWD + test, index=False)


def main():
    complexity = 2
    faces_vgg = f"/../data/03_primary/faces_vgg_{complexity}"
    train = '/../data/04_model_input/train'
    cv = '/../data/04_model_input/cv'
    test = '/../data/04_model_input/test'
    print(f'Train test split {datetime.datetime.now().strftime("%H:%M:%S")}')
    train_test_split(faces_vgg, train, cv, test)

if __name__ == '__main__':
    main()

 
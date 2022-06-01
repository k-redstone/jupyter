import pandas as pd
import numpy as np
import random

class MovielensData:
    def __init__(self, data_file_path, train_percent=0.8):
        self.data_file_path = data_file_path
        header = ['user_id', 'item_id', 'rating']
        df = pd.read_csv('data/movie_rating.csv', names=header)
        
        self.n_users = df.user_id.max()
        self.n_items = df.item_id.max()
        print ('Number of users = ' + str(self.n_users) + ' | Number of movies = ' + str(self.n_items))

        self.R = np.zeros((self.n_users+1, self.n_items+1))
        self.T = np.zeros((self.n_users+1, self.n_items+1))

        self.data = pd.DataFrame(df)
        random.seed(0)
        for line in self.data.itertuples():
            if (random.random() < train_percent):
                self.R[line[1], line[2]] = line[3]
            else:
                self.T[line[1], line[2]] = line[3] 

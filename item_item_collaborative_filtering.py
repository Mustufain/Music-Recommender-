import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import time
from sklearn.metrics import mean_squared_error


class MusicRecommender(object):
    """Item - Item based collaborative filtering."""

    def __init__(self):
        self.target = None
        self.datamatrix = None

    def create(self, data):
        self.target = data

    def get_unique_users(self):
        users = self.target.user_id.unique()
        return users

    def get_unique_songs(self):
        songs = self.target.song_id.unique()
        return songs

    def user_lookup(self):
        unique_users = self.get_unique_users()
        user_lookup = {unique_users[i]:
                       i for i in range(0, len(unique_users))}
        return user_lookup

    def song_lookup(self):
        unique_songs = self.get_unique_songs()
        song_lookup = {unique_songs[i]:
                       i for i in range(0, len(unique_songs))}
        return song_lookup

    def get_item_similarity(self, data_matrix):
        """Create item item similarity matrix of shape (songs, songs)
         based of cosine similarity."""
        item_similarity = pairwise_distances(data_matrix.T,
                                             metric='cosine',
                                             n_jobs=-1)
        return item_similarity

    def fit(self):
        """Creates a data matrix with shape (users, songs). The value of each
        cell of the datamatix is how many times user has listened that song.
        """
        data_matrix = np.zeros((len(self.get_unique_users()),
                                len(self.get_unique_songs())))
        user_lookup = self.user_lookup()
        song_lookup = self.song_lookup()
        for line in self.target.itertuples():
            data_matrix[user_lookup[line[1]], song_lookup[line[2]]] = line[3]
        self.datamatrix = data_matrix

    def predict_item(self, data_matrix):
        item_similarity = self.get_item_similarity(data_matrix)
        pred = data_matrix.dot(item_similarity) / np.array(
            [np.abs(item_similarity).sum(axis=1)])
        return pred

    def recommend(self, user_id):
        """Personalized top 10 recommendation for each user"""
        pred = self.predict_item(self.datamatrix)
        pred_transpose = pd.DataFrame(pred.T)
        user = self.user_lookup()[user_id]
        user_songs = pred_transpose[user]
        sorted_user_songs = user_songs.sort_values(ascending=False)
        top_10_songs_list = list(sorted_user_songs.index)[0:10]
        print ("----------Recommending top 10 songs for user_id " + str(
            user_id) + "----------")
        print (self.target.loc[top_10_songs_list]['song_name'])
        return self.evaluate_model(pred)

    def evaluate_model(self, pred):
        """Evaluate item item based collaborative filtering model"""
        error = []
        for i in range(0, self.datamatrix.shape[0]):
            error.append(mean_squared_error(self.datamatrix[i],  pred[i]))
        return round(np.mean(error), 4)


if __name__ == '__main__':
    start_time = time.time()
    song_df = pd.read_csv('data/songs.csv')
    song_df['song_name'] = song_df.apply(
        lambda row: str(row['title']) + " - " + str(row['artist_name']),
        axis=1)
    users = song_df['user_id'].unique()
    user_id = 'b80344d063b5ccb3212f76538f3d9e43d87dca9e'
    music = MusicRecommender()
    music.create(song_df)
    music.fit()
    mse = music.recommend(user_id)
    print ('Mean Squared Error: ' + str(mse))
    print("--- %s seconds ---" % (time.time() - start_time))

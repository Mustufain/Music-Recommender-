import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

""""
Personalized Recommendations for every user.
Recommendations for user[10]:

4074                              Be With You - Beyoncé
1426                  Relax - Frankie Goes To Hollywood
2234                            Home - Three Days Grace
1270                      Paradise City - Guns N' Roses
878       The Only Exception (Album Version) - Paramore
4069    Check On It - Beyoncé feat. Bun B and Slim Thug
661                   What They Found - Octopus Project
209                                Wet Blanket - Metric
3735             Oxford Comma (Album) - Vampire Weekend
388      Hippie Priest Bum-out (Edit) - LCD Soundsystem
"""


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

    def get_user_similarity(self, data_matrix):
        # similarities are obtained by subtracting distances from 1
        user_similarity = 1 - pairwise_distances(data_matrix, metric='cosine')
        return user_similarity

    def get_item_similarity(self, data_matrix):
        item_similarity = pairwise_distances(data_matrix.T, metric='cosine')
        return item_similarity

    def fit(self):
        data_matrix = np.zeros((len(self.get_unique_users()),
                                len(self.get_unique_songs())))
        user_lookup = self.user_lookup()
        song_lookup = self.song_lookup()
        for line in self.target.itertuples():
            data_matrix[user_lookup[line[1]], song_lookup[line[2]]] = line[3]
        self.datamatrix = data_matrix

    def predict_user(self, data_matrix):
        user_similarity = self.get_user_similarity(data_matrix)
        mean_user_target = data_matrix.mean(axis=1)
        target_diff = (data_matrix - mean_user_target[:, np.newaxis])
        pred = mean_user_target[:, np.newaxis]
        + user_similarity.dot(target_diff) / np.array(
            [np.abs(user_similarity).sum(axis=1)]).T
        return pred

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


if __name__ == '__main__':

    song_df = pd.read_csv('data/songs.csv')
    song_df['song_name'] = song_df.apply(
        lambda row: str(row['title']) + " - " + str(row['artist_name']),
        axis=1)
    users = song_df['user_id'].unique()
    user_id = users[10]
    music = MusicRecommender()
    music.create(song_df)
    music.fit()
    music.recommend(user_id)

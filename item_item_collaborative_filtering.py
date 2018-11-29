import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
# https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85
# https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/


class MusicRecommender(object):

    def __init__(self):

        triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
        songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'
        song_df_1 = pd.read_table(triplets_file, header=None)
        song_df_1.columns = ['user_id', 'song_id', 'listen_count']
        # Read song metadata
        song_df_2 = pd.read_csv(songs_metadata_file)
        # Merge the two dataframes above to create input dataframe for recommender systems
        song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(
            ['song_id']), on="song_id", how="left")
        song_df = song_df.head(10000)
        self.target = song_df
        self.target['song_name'] = self.target.apply(
            lambda row: str(row['title']) + " - " + str(row['artist_name']),
            axis=1)

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

    def create_user_songs_matrix(self):
        data_matrix = np.zeros((len(self.get_unique_users()),
                                len(self.get_unique_songs())))
        user_lookup = self.user_lookup()
        song_lookup = self.song_lookup()
        for line in self.target.itertuples():
            data_matrix[user_lookup[line[1]], song_lookup[line[2]]] = line[3]
        return data_matrix

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

    def recommend_item(self, data_matrix, user_id):
        """Personalized top 10 recommendation for each user"""
        pred = self.predict_item(data_matrix)
        pred_transpose = pd.DataFrame(pred.T)
        user = self.user_lookup()[user_id]
        user_songs = pred_transpose[user]
        sorted_user_songs = user_songs.sort_values(ascending=False)
        top_10_songs_list = list(sorted_user_songs.index)[0:10]
        print ("----------Recommending top 10 songs for user_id" + str(
            user_id) + "----------")
        print (self.target.loc[top_10_songs_list]['song_name'])


if __name__ == '__main__':

    user_id = "4bd88bfb25263a75bbdd467e74018f4ae570e5df"
    music = MusicRecommender()
    user_song_matrix = music.create_user_songs_matrix()
    music.recommend_item(user_song_matrix, user_id)

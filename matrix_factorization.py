import pandas as pd
import numpy as np
from sparsesvd import sparsesvd
from scipy.sparse import csc_matrix
from sklearn.metrics import mean_squared_error
import time


class MatrixFactorization(object):

    def __init__(self):
        self.target = None
        self.P = None
        self.Q = None
        self.W = None
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

    def fit(self):
        data_matrix = np.zeros((len(self.get_unique_users()),
                                len(self.get_unique_songs())))
        user_lookup = self.user_lookup()
        song_lookup = self.song_lookup()
        for line in self.target.itertuples():
            data_matrix[user_lookup[line[1]], song_lookup[line[2]]] = line[3]
        self.datamatrix = data_matrix
        U, S, Vt = self.computeSVD(data_matrix, 2)
        self.P = U
        self. W = S
        self.Q = Vt

    def computeSVD(self, urm, K):
        smat = csc_matrix(urm)
        U, s, Vt = sparsesvd(smat, K)
        dim = (len(s), len(s))
        S = np.zeros(dim, dtype=np.float32)
        for i in range(0, len(s)):
            S[i, i] = np.sqrt(s[i])
            U = csc_matrix(np.transpose(U), dtype=np.float32)
            S = csc_matrix(S, dtype=np.float32)
            Vt = csc_matrix(Vt, dtype=np.float32)
            return U, S, Vt

    def recommend(self, user):
        rightTerm = self.W * self.Q
        prod = self.P[user, :] * rightTerm
        estimatedListenCount = np.zeros(
            shape=(self.datamatrix.shape[0],
                   self.datamatrix.shape[1]), dtype=np.float16)
        estimatedListenCount[user, :] = prod.todense()
        recom = (-estimatedListenCount[user, :]).argsort()[:10]
        print ("----------Recommending top 10 songs for user_id " + str(
            self.target.loc[user].user_id) + "----------")
        for index in recom:
            print (self.target.loc[index].song_name)
        return self.evaluate_model(estimatedListenCount)

    def evaluate_model(self, pred):
        """Evaluate Matrix factorization  model"""
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
    user_id_index = 10
    print (
        'User b80344d063b5ccb3212f76538f3d9e43d87dca9e' +
        'has listened to the following songs')
    print (
        song_df.loc[
            song_df.user_id
            == 'b80344d063b5ccb3212f76538f3d9e43d87dca9e'].
        song_name)
    mf = MatrixFactorization()
    mf.create(song_df)
    mf.fit()
    mse = mf.recommend(user_id_index)
    print ('Mean Squared Error: ' + str(mse))
    print("--- %s seconds ---" % (time.time() - start_time))

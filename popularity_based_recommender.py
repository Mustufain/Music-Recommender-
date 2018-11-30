import pandas as pd

"""
Information about data:
Songs data source = 'https://static.turi.com/datasets/millionsong/10000.txt'
Meta data data source = 'https://static.turi.com/datasets/millionsong/song_data.csv'
Total 2000000 songs
First 10000 songs are selected
Unique users = 365
Unique songs = 5151
Users has listened to more than one song (multiple entries for each user) in
songs subset of 10000 songs

Note : These are not personalized recommendations. For every user,
recommendation will be same as it recommends top 10 trending songs.

Recommendation for user[10]:

7127                             Sehr kosmisch - Harmonia
9084                                         Undo - Björk
2068    Dog Days Are Over (Radio Edit) - Florence + Th...
9880                       You're The One - Dwight Yoakam
6774                              Revelry - Kings Of Leon
7115                                Secrets - OneRepublic
3613    Horn Concerto No. 4 in E flat K495: II. Romanc...
2717                       Fireflies - Charttraxx Karaoke
3485                             Hey_ Soul Sister - Train
8847                                   Tive Sim - Cartola
"""


class PopulairtyRecommender(object):
    """Recommending most popular/trending songs"""

    def __init__(self):
        self.data = None
        self.popularity_recommendations = None

    # Create the popularity based recommender system model
    def create(self, data):
        self.data = data

    def fit(self):
        # Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = self.data.groupby(
            'song_name')['user_id'].count().reset_index()
        train_data_grouped.columns = ['song_name', 'score']
        # Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(
            ['score', 'song_name'], ascending=[0, 1])
        # Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(
            ascending=0, method='first')
        # Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations
        # Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
        # Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        print ("----------Recommending top 10 songs for user_id " + str(
            user_id) + "----------")
        print (user_recommendations['song_name'])
        return user_recommendations


if __name__ == '__main__':

    song_df = pd.read_csv('data/songs.csv')
    song_df['song_name'] = song_df.apply(
        lambda row: str(row['title']) + " - " + str(row['artist_name']),
        axis=1)
    users = song_df['user_id'].unique()
    user_id = users[10]
    pr = PopulairtyRecommender()
    pr.create(song_df)
    pr.fit()
    pr.recommend(user_id)

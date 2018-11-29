import pandas as pd
from sklearn.model_selection import train_test_split

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
"""


class PopulairtyRecommender(object):
    """Recommending most popular/trending songs"""

    def __init__(self):
        self.train_data = None
        self.popularity_recommendations = None

    # Create the popularity based recommender system model
    def create(self, train_data):
        self.train_data = train_data

        # Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby(
            'song')['user_id'].count().reset_index()
        train_data_grouped.columns = ['song', 'score']
        # Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(
            ['score', 'song'], ascending=[0, 1])
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
        return user_recommendations


if __name__ == '__main__':

    song_df = pd.read_csv('data/songs.csv')
    users = song_df['user_id'].unique()
    train_data, test_data = train_test_split(
        song_df, test_size=0.20, random_state=0)
    pr = PopulairtyRecommender()
    pr.create(train_data)
    # user the popularity model to make some prediction
    user_id = users[10]
    user_recommendations = pr.recommend(user_id)
    print ("----------Recommending top 10 songs for user_id " + str(
        user_id) + "----------")
    print (user_recommendations['song'])

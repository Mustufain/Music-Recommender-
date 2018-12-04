import pandas as pd


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
    user_id = 'b80344d063b5ccb3212f76538f3d9e43d87dca9e'
    pr = PopulairtyRecommender()
    pr.create(song_df)
    pr.fit()
    pr.recommend(user_id)

import pandas as pd


def read_data(file):
    """read data from a file"""
    data = pd.read_csv('facebook-links.txt.anon', delimiter="\t", header=None)
    data.columns = ['user', 'user_friend_list', 'time']
    return data


if __name__ == "__main__":
    file = 'facebook-links.txt.anon'
    data = read_data(file)

import pandas as pd


def read_data(file):
    """read data from a file"""
    data = pd.read_csv('facebook-links.txt.anon', delimiter="\t", header=None)
    data.columns = ['user', 'user_friend_list', 'time']
    return data


def get_friends(user, data):
    """get friends for a given user"""
    setA = list(
        data.loc[data.user == user].user_friend_list.values)
    setB = list(
        data.loc[data.user_friend_list == user].user
        .values)
    friends = list(set(set(setA).union(setB)))
    return friends


def get_friends_of_friend(friends, data):
    """Find friends of friends for a given user"""
    friends_of_friends = []
    for friend in friends:
        friend_list = get_friends(friend, data)
        friends_of_friends.append(friend_list)
    return sum(friends_of_friends, [])


def get_common_friends(user, friends, friends_of_friends, data):
    """Get common friends between a user and friends of friends"""
    common_friends_list = {}
    friends_set = set(friends)
    for friend_of_friend in list(set(friends_of_friends)):
        if int(friend_of_friend) != user and friend_of_friend not in friends:
            friend_of_friend_list = get_friends(friend_of_friend, data)
            score = len(list(friends_set.intersection(friend_of_friend_list)))
            if score in common_friends_list:
                common_friends_list[score].append(friend_of_friend)
            else:
                common_friends_list[score] = [friend_of_friend]
    return common_friends_list


def get_top_friends(common_friends_list):
    """Get top 4 friends for a given user"""
    n = 4
    top_n_users = []
    top_scores = sorted(common_friends_list, reverse=True)
    for score in top_scores:
            top_n_users.append(sorted(common_friends_list[score]))
    top_n_users = sum(top_n_users, [])
    return top_n_users[:n]


if __name__ == "__main__":
    file = 'facebook-links.txt.anon'
    user_list = [20341, 33722, 35571, 25017]
    data = read_data(file)
    for user in user_list:
        friends = get_friends(user, data)
        friends_of_friends = get_friends_of_friend(friends, data)
        common_friends_list = get_common_friends(user,
                                                 friends,
                                                 friends_of_friends,
                                                 data)
        recommendation = get_top_friends(common_friends_list)
        print ("Top 4 Friends suggested to user " +
               str(user) +
               " are " +
               str(recommendation))

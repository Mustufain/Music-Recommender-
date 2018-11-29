import pandas as pd


def read_data(file):
    """read data from a file"""
    data = pd.read_csv('facebook-links.txt.anon', delimiter="\t", header=None)
    data.columns = ['user', 'user_friend_list', 'time']
    return data


def get_friends(user, data):
    """get friends for a given user"""
    friends = list(
        data.loc[data.user == user].user_friend_list.values)
    print (friends)
    return friends


def get_friends_of_friend(friends, data):
    """Find friends of friends for a given user"""
    friends_of_friends = {}
    for friend in friends:
        friend_list = list(
            data.loc[data.user == friend].user_friend_list.values)
        friends_of_friends[friend] = friend_list
    return friends_of_friends


def get_common_friends(friends, friends_of_friends):
    """Get common friends between a user and friends of friends"""
    common_friends_list = []
    friends_set = set(friends)
    for key, value in friends_of_friends.items():
        result = (key, list(friends_set.intersection(value)))
        common_friends_list.append(result)
    return common_friends_list


def get_influence_score(common_friends_list, data):
    """Get influence score for a user"""
    no_of_friends_list = []
    influence_score_list = []
    for friend in common_friends_list:
        for common_friend in friend[1]:
            no_of_friends = len(list(
                data.loc[data.user == common_friend].user_friend_list.values))
            no_of_friends_list.append(no_of_friends)
        influence_score = float(1/sum(no_of_friends_list))
        result = (friend, influence_score)
        influence_score_list.append(result)
    return influence_score_list


def get_top_4_friends(influence_score_list):
    """Get top 4 friends for a given user"""
    sorted_influence_score_list = sorted(influence_score_list,
                                         key=lambda tup: tup[1], reverse=True)
    top_4 = sorted_influence_score_list[:4]
    top_4_users = []
    for user in top_4:
        top_4_users.append(user[0])
    return top_4_users


if __name__ == "__main__":
    file = 'facebook-links.txt.anon'
    user_list = [20341, 33722, 35571, 25017]
    data = read_data(file)
    for user in user_list:
        friends = get_friends(user, data)
        friends_of_friends = get_friends_of_friend(friends, data)
        common_friends_list = get_common_friends(friends, friends_of_friends)
        get_influence_score(common_friends_list, data)
        recommendation = get_top_4_friends(common_friends_list)
        print ("Top 4 Friends suggested to user " +
               str(user) +
               " are " +
               str(recommendation))

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
    return friends


def get_friends_of_friend(friends, data):
    """Find friends of friends for a given user"""
    friends_of_friends = []
    for friend in friends:
        friend_list = list(
            data.loc[data.user == friend].user_friend_list.values)
        friends_of_friends.append(friend_list)
    return sum(friends_of_friends, [])


def get_common_friends(user, friends, friends_of_friends, data):
    """Get common friends between a user and friends of friends"""
    common_friends_list = []
    friends_set = set(friends)
    for friend_of_friend in list(set(friends_of_friends)):
        if int(friend_of_friend) != user and friend_of_friend not in friends:
            friend_of_friend_list = list(
                data.loc[data.user ==
                         friend_of_friend].user_friend_list.values)
            result = (friend_of_friend, len(list(
                friends_set.intersection(friend_of_friend_list))))
            common_friends_list.append(result)
    return common_friends_list


def get_top_4_friends(common_friends_list):
    """Get top 4 friends for a given user"""
    sorted_common_friends_list = sorted(common_friends_list,
                                        key=lambda tup: tup[1], reverse=True)
    top_4 = sorted_common_friends_list[:4]
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
        common_friends_list = get_common_friends(user,
                                                 friends,
                                                 friends_of_friends,
                                                 data)
        recommendation = get_top_4_friends(common_friends_list)
        print ("Top 4 Friends suggested to user " +
               str(user) +
               " are " +
               str(recommendation))
        break

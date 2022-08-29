import pandas as pd
import random
import os


def load_sessions(path="../data"):
    """
    load and merge the train_session-x.csv data
    :param data_path:
    :return: dictionary {session_id: [item_id1,item_id2,...]}
    """
    df_sessions = pd.read_csv(os.path.join(path,"train_sessions-1.csv"))
    for i in range(2,6):
        file = "train_sessions-" + str(i) + ".csv"
        df_sessions.append = pd.read_csv(os.path.join(path, file))
    df_sessions.drop("date",axis=1,inplace=True)
    dict_sessions=dict()
    for row in df_sessions.itertuples(index=False):
        if row.session_id not in dict_sessions:
            dict_sessions[row.session_id]=list()
        dict_sessions[row.session_id].append(row.item_id)
    return dict_sessions


def train_test_split(data_path="../data",ratio=0.7,random_state=1,y_format="dict"):
    """
    :param data_path: path to datadir where train_sessions-x.csv and train_purchases are stored
    :param ratio: ratio for train and test split
    :param random_state: random_state (default 1)
    :param y_format: (choose a format for y-values aka purchase data, default is "dict", but "df" might be useful sometimes)

    :return: train_sessions, test_sessions, train_purchases, test_purchases
    or in other words:
    :return: X_train,X_test,y_train,y_test

    note: test_purchase (aka y_test) is probably not needed afterwards, because evaluation dont need them
    """


    #load train_sessions
    dict_sessions=load_sessions(data_path)

    #split sessions into train and test
    session_ids=sorted(list(dict_sessions))
    random.seed(random_state)
    #train
    train_ids=random.sample(session_ids,int(len(session_ids)*ratio))
    train_sessions={train_id:dict_sessions[train_id] for train_id in train_ids}
    #test
    test_ids=[id for id in session_ids if id not in train_sessions]
    test_sessions={test_id:dict_sessions[test_id] for test_id in test_ids}

    #create train_purchase and test_purchase
    #load
    df_purchases = pd.read_csv(os.path.join(data_path, "train_purchases.csv"))
    df_purchases.drop("date", axis=1, inplace=True)
    df_purchases.set_index("session_id", inplace=True)
    #train
    train_purchases = df_purchases.loc[train_ids]
    train_purchases.sort_index(inplace=True)
    #test
    test_purchases = df_purchases.loc[test_ids]
    test_purchases.sort_index(inplace=True)

    if y_format=="df":
        return train_sessions, test_sessions, train_purchases, test_purchases
    else:
        #convert purchases into dicts:
        train_purchases=train_purchases["item_id"].to_dict()
        test_purchases=test_purchases["item_id"].to_dict()
        return train_sessions, test_sessions, train_purchases, test_purchases

def train_test_split_challenge(path="../data",challenge="leaderboard",y_format="dict"):
    """
    Function which uses all data for training, and for test the corresponding data set
    This function should be used for testing the models on the website, it cannot be used with our evaluation-functions
    :param challenge: choose <"leaderboard","final">
    :return: X_train, X_test, y_train
    """
    #loaded full training data into train_sessions
    train_sessions, placeholder, train_purchases, placeholder1=train_test_split(ratio=1,y_format=y_format)

    #load test data
    filename="test_" + challenge + "_sessions.csv"
    df_test = pd.read_csv(os.path.join(path,filename))
    df_test.drop("date",axis=1,inplace=True)
    test_sessions=dict()
    for row in df_test.itertuples(index=False):
        if row.session_id not in test_sessions:
            test_sessions[row.session_id]=list()
        test_sessions[row.session_id].append(row.item_id)
    return train_sessions, test_sessions, train_purchases

if __name__ == "__main__":
    train_sessions, test_sessions, train_purchases, test_purchases=train_test_split(y_format="df")
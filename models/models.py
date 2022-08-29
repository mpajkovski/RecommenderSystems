import pandas as pd
import numpy as np
import scipy.sparse
from functools import lru_cache

from time import gmtime, strftime


# -----Basis model-----
# Please inherete from it

class BasisRecommender:
    model = None  # can be anything (also a new class), whatever is useful for your Recommender

    def train(self, X_train, y_train):
        """
        Should train a model using data and assign trained_model to self.model
        """
        NotImplementedError("train method not implemented")

    def predict(self, X_test) -> pd.DataFrame:
        """
        Should use pretrained model (from self.model) to predict 100 items per session_id
        Output should be a pd.dataframe with 3 columns:
        col = ["session_id", "item_id", "rank"]
        """
        raise NotImplementedError("predict method not implemented")


# ------Baseline Model--------
# This is a super simple model, every more advanced model just outperform this one

class BaselineRecommender(BasisRecommender):

    def train(self, X_train, y_train):
        # get the most purchased items in the training data (regardless of session history)
        a = y_train.reset_index().groupby("item_id").count().sort_values(
            by="session_id", ascending=False)
        # because its a very simple model, here model will just be a series
        self.model = a.reset_index()["item_id"].iloc[:100]
        # notification
        print("Baseline model trained!")

    def predict(self, X_test) -> pd.DataFrame:
        row_list = []
        for key in X_test.keys():
            for i in range(100):
                row = (key, self.model.iloc[i], i + 1)
                row_list.append(row)
        return pd.DataFrame(row_list, columns=["session_id", "item_id", "rank"])


# ------Content-Based Recommendation Model--------
# Recommends the most similar item (according to features) of the top N items from the baseline model

class ContentBasedRecommendation(BasisRecommender):

    def __init__(self,
                 take_top_n=2000,
                 num_return=100,
                 path_item_feature_df="../data/item_features.csv",
                 candidate_only=False):

        self.take_top_n = take_top_n  # only recommends from top items (because most of them are shit)
        self.num_return = num_return  # number of items that will be recommended
        self.model = list()  # tuple of item_id and prototype vector (?)
        self.unique_features = None
        self.df_item_features = pd.read_csv(path_item_feature_df)
        self.candidate_only = candidate_only

    @lru_cache(maxsize=1024)
    def get_features_from_item(self, item):
        cur_features = self.df_item_features[self.df_item_features["item_id"] == item]
        cur_features = cur_features[['feature_category_id', 'feature_value_id']] \
            .astype(str).agg('-'.join, axis=1).unique().tolist()

        return cur_features

    @staticmethod
    def simple_error_function(user_items_features, top_item_feature):
        err = 0
        for item in user_items_features:
            for feature in item:
                if feature not in top_item_feature:
                    err += 1
                else:
                    err -= 10

        return err

    def train(self, X_train, y_train):
        # find top 2000 items
        df_sorted = y_train.reset_index().groupby("item_id").count().sort_values(by="session_id", ascending=False)
        if self.candidate_only:
            # filter df_sorted by items in candidate list
            df_candidate_items = pd.read_csv("./../data/candidate_items.csv")
            df_sorted = pd.merge(df_sorted, df_candidate_items, left_on="item_id", right_on="item_id")

        items_top_n = df_sorted.reset_index()["item_id"].iloc[:self.take_top_n].tolist()

        df_if_top_n = self.df_item_features[self.df_item_features["item_id"].isin(items_top_n)]
        self.unique_features = df_if_top_n[['feature_category_id', 'feature_value_id']] \
            .astype(str).agg('-'.join, axis=1).unique()

        # create list of tuples of the top 500"
        self.model = list()
        for item in items_top_n:
            # create tuple with (item_id, list of features)
            cur_features = self.get_features_from_item(item)
            self.model.append((item, cur_features))

    def predict(self, X_test) -> pd.DataFrame:
        result_df = list()
        for num_i, u_id in enumerate(X_test):
            user_items_features = list()
            for item in X_test[u_id]:
                # create same list of features
                cur_features = self.get_features_from_item(item)
                user_items_features.append(cur_features)

            res_list = list()
            for top_tuple in self.model:
                # calculate MSE between user_items_features and top_tuple's features
                mse = self.simple_error_function(user_items_features, top_tuple[1])
                res_list.append((top_tuple[0], mse))

            # sort and append self.num_return
            res_list = sorted(res_list, key=lambda tup: tup[1])[:self.num_return]
            res_list = [[u_id, tup[0], rank + 1] for rank, tup in enumerate(res_list)]
            result_df += res_list

        col = ["session_id", "item_id", "rank"]

        return pd.DataFrame(data=result_df, columns=col)


# ------Session neighborhood Model--------
# This model counts items that appear together in sessions and recommends the items that appear together most often

class SessionSimilarityRecommender(BasisRecommender):

    def train(self, X_train, y_train):
        # combine X and y
        dict_sessions = X_train.copy()
        for key, value in y_train.items():
            if key in dict_sessions:
                dict_sessions[key].append(value)
            else:
                dict_sessions[key] = [value]

        #TODO: make this call nicer
        df_candidate_items = pd.read_csv("../data/candidate_items.csv")
        
        array_candidates = df_candidate_items["item_id"]

        # sparse matrix item item
        sm_item_item = scipy.sparse.lil_matrix((28144,28144), dtype=np.int8)

        #iterate through all sessions
        for session, items in dict_sessions.items():
            #within a session compare all items with each other
            for item_2 in items:
                if type(item_2) is not int:
                    break
                #only write items to the second dimension if they are possible candidates for recommendation
                if item_2 in array_candidates.values:
                    #continuation of comparing all items with each other
                    for item_1 in items:
                        #dont recommend itself
                        if item_1 != item_2:
                            #dim_1: searched for item, dim_2: recommender candidate
                            sm_item_item[item_1, item_2] += 1
        self.model = sm_item_item
        # notification
        print("Session similarity model trained!")


    def predict(self, X_test) -> pd.DataFrame:
        dict_test_sessions = X_test.copy()

        df_recommendations = pd.DataFrame(columns=["session_id", "item_id", "rank"])
        
        # iterate through all test sessions   
        for session, items in dict_test_sessions.items():
            # save recommendations for this session (28144 is the total amount of available items)
            recommendations = np.zeros(28144)
            # iterate over items within a session
            for item in items:
                row = self.model.getrow(item).toarray()
                row = np.array(row[0])
                # add recommendations per item to recommendations per session
                recommendations += row
            # create the three columns session_id, recommended_item_id, rank
            item_ids = recommendations.nonzero()[0]
            values = [recommendations[i] for i in item_ids]
            arr_session = [session for i in range(0,len(values))]

            # create intermediate df
            df_session_rec = pd.DataFrame([arr_session, item_ids, values]).transpose()
            df_session_rec.columns = df_recommendations.columns
            df_session_rec.sort_values(by="rank", ascending=False, inplace=True)

            # cut top 100 recommendations
            df_session_rec = df_session_rec.head(100)
            new_rank = list(range(1, len(df_session_rec.index) + 1))
            df_session_rec["rank"] = new_rank
            df_recommendations = df_recommendations.append(df_session_rec)
        return df_recommendations



# ------Item Item Model--------
# This model ...

class Item_Item_recommender(BasisRecommender):

    def __init__(self, df_train):

        # model initialisation, assigns the train dataset
        self.df_train = df_train
        self.SimilarityMatrix = dict()

    def ComputeSimilarity(self):
        # This function compute the cosine similarity of items
        N = defaultdict(int)
        for user, items in self.df_train.items():

            itemset = set(items)
            for i in itemset:
                N[i] += 1

            for i in items:
                self.SimilarityMatrix.setdefault(i, dict())

                for j in items:
                    if i == j:
                        continue
                    self.SimilarityMatrix[i].setdefault(j, 0)
                    self.SimilarityMatrix[i][j] += 1


        for i, related_items in self.SimilarityMatrix.items():
            for j, cij in related_items.items():
                self.SimilarityMatrix[i][j] = cij / math.sqrt(N[i]*N[j])



    def train(self):
        self.ComputeSimilarity()


    def MakeRecommendations(self, user, N, K):

        recommendations = dict()
        #First get the list of user's favorite items
        items = self.df_train[user]
        point = 1
        for item in items:
            # For each user's favorite item, find the K most similar items in the item similarity matrix
            for i, sim in sorted(self.SimilarityMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if i in items:
                    continue  # If it is repeated with the user's favorite item, skip it directly
                recommendations.setdefault(i, 0.)
                recommendations[i] += sim * point
            point += 1.5
        # Arrange in reverse order according to the similarity of the recommended items,
        # and then recommend the top N items to the user
        return sorted(recommendations.items(), key=itemgetter(1), reverse=True)[:N]

    def predict(self, X_test):
        results = []
        for i in X_test:
            ii = 1
            recommendations = self.MakeRecommendations(i, 100, 2000)
            for j,k in recommendations:
                if(ii > 100):
                    break
                if(j in candidate):
                    results.append([str(i),str(j),str(ii)])
                    ii = ii+1

        return pd.DataFrame(results,columns = ['session_id','item_id','rank'])#.to_csv('sampleSubmission.csv',index=False)



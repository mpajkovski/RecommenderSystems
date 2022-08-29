import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_evaluation(result_df: pd.DataFrame = pd.DataFrame(),
                   result_df_path: str = None,
                   purchases_df_path: str = "../data/train_purchases.csv",
                   print_res: bool = False) -> dict:
    """
    Either hand over a dataframe of the results OR a path where the results can be found.
    run_evaluation will take the dataframe if both is handed over.
    If your data is organized differently you can specify where to find the train_purchases.csv
    """

    if result_df.size == 0 and result_df_path:
        result_df = pd.read_csv(result_df_path)
    elif result_df.size == 0 and not result_df_path:
        raise ValueError("Neither a Dataframe nor a path to a path to a Dataframe was specified!")
    # else: just use result_df as provided

    purchases_df = pd.read_csv(purchases_df_path)

    # calc position per session and calc metrics
    # print(result_df.head())
    # print(purchases_df.head())

    sessions = result_df["session_id"].unique()
    rank_list = list()

    num_not_included = 0
    for s in sessions:
        item_series = purchases_df[purchases_df["session_id"] == s]
        if item_series.size > 0:  # if session in purchases_df
            item = item_series["item_id"].item()  # get purchased item
        else:
            print(f"Error: Could not find session {s} in \"{purchases_df_path}\"")
            continue

        rank_series = result_df.query(f"session_id == {s} & item_id == {item}")
        if rank_series.size > 0:  # if item in result list
            rank = rank_series["rank"].item()
        else:  # if it's not included in the result list
            rank = np.inf
            num_not_included += 1

        rank_list.append(rank)

    MRR = np.mean([1/rank for rank in rank_list])
    mean_rank = np.mean([rank for rank in rank_list if rank != np.inf])
    std_rank = np.std([rank for rank in rank_list if rank != np.inf])

    accuracy = (len(rank_list)-num_not_included)/len(rank_list)
    accuracy_100 = sum([1 for rank in rank_list if rank <= 100])/len(rank_list)
    accuracy_50 = sum([1 for rank in rank_list if rank <= 50])/len(rank_list)
    accuracy_10 = sum([1 for rank in rank_list if rank <= 10])/len(rank_list)

    eval_results = dict({
        "MRR": MRR,
        "mean_position": mean_rank,
        "std_position": std_rank,
        "accuracy": accuracy,
        "accuracy_100": accuracy_100,
        "accuracy_50": accuracy_50,
        "accuracy_10": accuracy_10
    })

    if print_res:
        print(json.dumps(eval_results, indent=2))

    return eval_results


if __name__ == "__main__":
    col = ["session_id", "item_id", "rank"]
    data = [
        [3, 1, 1],
        [3, 2, 2],
        [3, 3, 3],
        [3, 15085, 4],
        [3, 5, 5],
        [3, 6, 6],
        [3, 7, 7],
        [3, 8, 8],
        [3, 9, 9],

        [13, 1, 1],
        [13, 18626, 2],
        [13, 3, 3],
        [13, 4, 4],
        [13, 5, 5],
        [13, 6, 6],
        [13, 7, 7],
        [13, 8, 8],
        [13, 9, 9],

        [18, 1, 1],
        [18, 2, 2],
        [18, 3, 3],
        [18, 4, 4],
        [18, 24911, 5],
        [18, 6, 6],
        [18, 7, 7],
        [18, 8, 8],
        [18, 9, 9],

        [19, 1, 1],
        [19, 2, 2],
        [19, 3, 3],
        [19, 4, 4],
        [19, 5, 5],
        [19, 6, 6],
        [19, 7, 7],
        [19, 12534, 8],
        [19, 9, 9],

        [24, 1, 1],
        [24, 2, 2],
        [24, 3, 3],
        [24, 4, 4],
        [24, 5, 5],
        [24, 6, 6],
        [24, 7, 7],
        [24, 8, 8],
        [24, 9, 9],
    ]

    a = [
        [18, 1, 1],
        [18, 2, 2],
        [18, 3, 3],
        [18, 4, 4],
        [18, 5, 5],
        [18, 6, 6],
        [18, 7, 7],
        [18, 8, 8],
        [18, 9, 9],

        [1, 1, 1]
    ]

    run_evaluation(pd.DataFrame(columns=col, data=data))

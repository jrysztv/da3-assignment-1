import pandas as pd
import numpy as np


# Extract the mean and standard deviation of the train and test scores for each alpha
def retrieve_cv_summary(fit_results, ols_cv_list=None, ols_model_number=None):
    cv_results = fit_results.cv_results_
    param_keys = list(fit_results.param_grid.keys())
    param_values = {
        key: fit_results.cv_results_[f"param_{key}"].data for key in param_keys
    }

    mean_train_scores = cv_results["mean_train_score"]
    std_train_scores = cv_results["std_train_score"]
    mean_test_scores = cv_results["mean_test_score"]
    std_test_scores = cv_results["std_test_score"]
    rank_test_scores = cv_results["rank_test_score"]
    estimator_name = type(fit_results.estimator).__name__

    # Create a DataFrame to store the results
    cv_metrics = pd.DataFrame(
        {
            "estimator": [estimator_name] * len(mean_train_scores),
            **param_values,
            "mean_train_score": np.abs(mean_train_scores),
            "std_train_score": std_train_scores,
            "mean_test_score": np.abs(mean_test_scores),
            "std_test_score": std_test_scores,
            "rank_test_score": rank_test_scores,
        }
    )

    if ols_cv_list and ols_model_number:
        ols_mean_train_score = np.array(ols_cv_list[ols_model_number]["rmse"]).mean()
        ols_mean_test_score = np.array(
            ols_cv_list[ols_model_number]["rmse_test"]
        ).mean()
        ols_std_train_score = np.array(ols_cv_list[ols_model_number]["rmse"]).std()
        ols_std_test_score = np.array(ols_cv_list[ols_model_number]["rmse_test"]).std()
        ols_rank_test_score = 1

        ols_metrics = {
            "estimator": [f"OLS Model {ols_model_number}"],
            "mean_train_score": [ols_mean_train_score],
            "std_train_score": [ols_std_train_score],
            "mean_test_score": [ols_mean_test_score],
            "std_test_score": [ols_std_test_score],
            "rank_test_score": [ols_rank_test_score],
        }

        ols_metrics_df = pd.DataFrame(ols_metrics)
        cv_metrics = pd.concat([cv_metrics, ols_metrics_df])
        cv_metrics["mean_diff_test_train"] = (
            cv_metrics["mean_test_score"] - cv_metrics["mean_train_score"]
        )

    return cv_metrics.sort_values("rank_test_score")


def summarize_multiple_cv_results(
    results_list, ols_cv_list=None, ols_model_number=None
):
    all_summaries = []
    for i, result in enumerate(results_list):
        if i == len(results_list) - 1:
            summary = retrieve_cv_summary(result, ols_cv_list, ols_model_number)
        else:
            summary = retrieve_cv_summary(result)
        all_summaries.append(summary)

    summaries_df = pd.concat(all_summaries, ignore_index=True)
    summaries_df = summaries_df.sort_values("rank_test_score")
    # Reorder the columns as requested
    ordered_columns = [
        "estimator",
        "mean_test_score",
        "mean_train_score",
        "mean_diff_test_train",
        "std_test_score",
        "std_train_score",
        "rank_test_score",
    ]
    ordered_columns += [
        col for col in summaries_df.columns if col not in ordered_columns
    ]

    summaries_df = summaries_df[ordered_columns]

    return summaries_df

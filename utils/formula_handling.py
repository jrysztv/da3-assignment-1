import pandas as pd
import statsmodels.formula.api as smf

# from sklearn.model_selection import KFold
from statsmodels.tools.eval_measures import rmse


def create_formula(df, target, features, interaction_terms=None):
    """
    Create a formula string for patsy.dmatrices or statsmodels.

    Parameters:
    target (str): The target variable.
    features (list): A list of feature names.
    interaction_terms (list, optional): A list of interaction terms.

    Returns:
    str: A formula string.
    """
    # Infer categorical variables from the dataframe
    categorical_vars = df.select_dtypes(include=["category"]).columns.tolist()

    # Wrap categorical variables in "C()"
    features = [
        f"C({feature})" if feature in categorical_vars else feature
        for feature in features
    ]

    formula = f"{target} ~ " + " + ".join(features)
    if interaction_terms:
        formula += " + " + " + ".join(interaction_terms)
    return formula


def cv_reg(formula, df, kfold, testdf, robustse=None):
    regression_list = []
    predicts_on_test = []
    rsquared = []
    rmse_list = []
    rmse_list_test = []

    # Calculating OLS for each fold

    for train_index, test_index in kfold.split(df):
        df_train, df_test = df.iloc[train_index, :], df.iloc[test_index, :]
        if robustse is None:
            model = smf.ols(formula, data=df_train).fit()
        else:
            model = smf.ols(formula, data=df_train).fit(cov_type=robustse)
        regression_list += [model]
        predicts_on_test += [model.predict(df_test)]
        rsquared += [model.rsquared]

        rmse_tr = pd.concat(
            [df_train["price"], model.predict(df_train)],
            axis=1,
            keys=["price", "predicted"],
        )
        rmse_tr = rmse_tr[~rmse_tr.isna().any(axis=1)]

        rmse_te = pd.concat(
            [df_test["price"], model.predict(df_test)],
            axis=1,
            keys=["price", "predicted"],
        )
        rmse_te = rmse_te[~rmse_te.isna().any(axis=1)]

        rmse_list += [rmse(rmse_tr["price"], rmse_tr["predicted"], axis=0)]
        rmse_list_test += [rmse(rmse_te["price"], rmse_te["predicted"], axis=0)]
    nvars = model.df_model

    return {
        "regressions": regression_list,
        "test_predict": predicts_on_test,
        "r2": rsquared,
        "rmse": rmse_list,
        "rmse_test": rmse_list_test,
        "nvars": nvars,
    }


def summarize_cv(cvlist, stat="rmse"):
    result = pd.DataFrame(
        {"Model" + str(x + 1): cvlist[x][stat] for x in range(len(cvlist))}
    )
    result["Resample"] = ["Fold" + str(x + 1) for x in range(len(cvlist[0]["rmse"]))]
    result = result.set_index("Resample")
    result = pd.concat([result, pd.DataFrame(result.mean(), columns=["Average"]).T])
    return result

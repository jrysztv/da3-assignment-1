import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_predictions_vs_actuals(
    estimator_cv, X_holdout, y_holdout, ax=None, title="Predicted vs Actual Prices"
):
    """
    Plot the predicted vs actual prices for a given estimator.

    Parameters:
    estimator_cv (GridSearchCV): The fitted estimator cross-validation object to use for predictions.
    X_holdout (array-like): The holdout set features.
    y_holdout (array-like): The holdout set actual prices.
    ax (matplotlib.axes._axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created.
    title (str): The title of the plot.

    Returns:
    matplotlib.axes._axes.Axes: The axes object of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Predict on the holdout set
    estimator = estimator_cv.best_estimator_
    predictions = estimator.predict(X_holdout)

    # Plot the predicted vs actual prices
    sns.scatterplot(x=predictions, y=y_holdout, ax=ax, alpha=0.1)
    ax.plot(
        [y_holdout.min(), y_holdout.max()],
        [y_holdout.min(), y_holdout.max()],
        "k--",
        lw=2,
    )
    ax.set_xlabel("Predicted Prices")
    ax.set_ylabel("Actual Prices")
    ax.set_title(title)

    return ax


def plot_multiple_predictions_vs_actuals(
    cv_list, X_holdout, y_holdout, n_rows=2, main_title="Predicted vs Actual Prices"
):
    """
    Plot the predicted vs actual prices for multiple estimators.

    Parameters:
    cv_list (list): List of fitted estimator cross-validation objects.
    X_holdout (array-like): The holdout set features.
    y_holdout (array-like): The holdout set actual prices.
    n_rows (int): Number of rows in the plot grid.
    main_title (str): The main title of the plot.

    Returns:
    None
    """
    n_cols = (len(cv_list) + n_rows - 1) // n_rows
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 4)
    )
    axes = axes.flatten()

    for ax, estimator_cv, i in zip(axes, cv_list, range(len(cv_list))):
        estimator_name = type(estimator_cv.best_estimator_).__name__
        title = f"Model {i + 1}: {estimator_name}"
        plot_predictions_vs_actuals(
            estimator_cv, X_holdout, y_holdout, ax=ax, title=title
        )

    for i in range(len(cv_list), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1.005])
    plt.show()


def get_feature_importance(estimator):
    """
    Extracts feature importances from a fitted GridSearchCV estimator.
    Returns a dictionary with feature names and their importance, or None if not available.
    """
    model_name = estimator.best_estimator_.__class__.__name__  # Get model class name
    try:
        importance = estimator.best_estimator_.feature_importances_
        return {"model": model_name, "importances": importance}
    except AttributeError:
        return None  # If the model does not support feature_importances_


def plot_single_feature_importance(
    feature_names, importances, model_name, top_n=None, ax=None
):
    """
    Plots feature importance for a single model using Seaborn.

    Parameters:
    - feature_names: List of feature names.
    - importances: Feature importance scores.
    - model_name: Name of the model.
    - top_n: Number of top features to display (optional).
    - ax: Seaborn axis (optional, if provided, will plot there).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))

    # Sort by importance within the model
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[sorted_indices]
    sorted_importances = importances[sorted_indices]

    if top_n:
        sorted_features = sorted_features[:top_n]
        sorted_importances = sorted_importances[:top_n]

    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, color="royalblue")
    ax.set_title(f"{model_name} Feature Importances")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")


def plot_feature_importances(estimators, X, top_n=None, n_rows=None):
    """
    Plots feature importances for multiple estimators in subplots using Seaborn.

    Parameters:
    - estimators: List of fitted GridSearchCV objects.
    - X: DataFrame of features (used to extract feature names).
    - top_n: Number of top features to display (optional).
    - n_rows: Number of subplot rows (optional, automatically determined if None).
    """
    feature_names = np.array(X.design_info.column_names)  # Extract feature names

    # Filter out models without feature_importances_
    importance_data = [get_feature_importance(est) for est in estimators]
    importance_data = [data for data in importance_data if data]  # Remove None values

    num_estimators = len(importance_data)
    if num_estimators == 0:
        print("No estimators have feature importances to display.")
        return

    if n_rows is None:
        n_rows = num_estimators  # Default to one row per estimator

    n_cols = (num_estimators + n_rows - 1) // n_rows  # Ensure all plots fit in the grid

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True
    )

    axes = np.array(axes).reshape(-1)  # Flatten for easy iteration

    for ax, data in zip(axes, importance_data):
        plot_single_feature_importance(
            feature_names, data["importances"], data["model"], top_n, ax=ax
        )

    # Hide unused subplots
    for i in range(len(importance_data), len(axes)):
        fig.delaxes(axes[i])

    plt.show()

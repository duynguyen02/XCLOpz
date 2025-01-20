from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


class XCLOpz:
    """
    A class for automated training and optimization of multiple gradient boosting models
    (XGBoost, LightGBM, and CatBoost) for regression tasks.

    This class provides functionality for:
    - Data preparation and splitting
    - Hyperparameter optimization
    - Model training and evaluation
    - Feature importance visualization
    - Model persistence

    Parameters
    ----------
    random_state : int, optional (default=42)
        Random seed for reproducibility
    use_gpu : bool, optional (default=False)
        Whether to use GPU acceleration for training

    Attributes
    ----------
    models : dict
        Dictionary storing the trained models
    best_params : dict
        Dictionary storing the best parameters for each model
    scores : dict
        Dictionary storing evaluation metrics for each model
    """

    def __init__(self, random_state=42, use_gpu=False):
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.models = {}
        self.best_params = {}
        self.scores = {}

    def prepare_data_v2(
        self,
        df: pd.DataFrame,
        y_col: str,
        x_cols: List[str],
        test_size: float = 0.2,
        copy_df: bool = False,
    ):
        if copy_df:
            df = df.copy()

        X = df[[x_cols]]
        y = df[y_col]

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=test_size, random_state=self.random_state
        )

        print("Dataset dimensions:")
        print(f"Train: {self.X_train.shape}")
        print(f"Validation: {self.X_val.shape}")
        print(f"Test: {self.X_test.shape}")

        return self

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        drop_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        copy_df: bool = False,
    ):
        """
        Prepare and split the dataset into training, validation, and test sets.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing features and target
        target_col : str
            Name of the target column
        drop_cols : List[str], optional
            List of columns to drop from the dataset
        test_size : float, optional (default=0.2)
            Proportion of the dataset to include in the test and validation splits
        copy_df : bool, optional (default=False)
            Whether to create a copy of the input DataFrame

        Returns
        -------
        self : object
            Returns the instance itself
        """
        if copy_df:
            df = df.copy()

        if drop_cols is None:
            drop_cols = []
        drop_cols.append(target_col)

        X = df.drop(drop_cols, axis=1)
        y = df[target_col]

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=test_size, random_state=self.random_state
        )

        print("Dataset dimensions:")
        print(f"Train: {self.X_train.shape}")
        print(f"Validation: {self.X_val.shape}")
        print(f"Test: {self.X_test.shape}")

        return self

    def train_xgboost(self):
        """
        Train and optimize an XGBoost model using grid search.

        The method performs a grid search over specified hyperparameters and selects
        the best model based on validation RMSE. Early stopping is used to prevent
        overfitting.

        Returns
        -------
        self : object
            Returns the instance itself
        """
        print("\nTraining XGBoost...")

        param_grid = {
            "n_estimators": [1000],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "gamma": [0, 0.1],
        }

        best_rmse = float("inf")
        best_params = None
        best_model = None

        for max_depth in param_grid["max_depth"]:
            for learning_rate in param_grid["learning_rate"]:
                for min_child_weight in param_grid["min_child_weight"]:
                    for subsample in param_grid["subsample"]:
                        for colsample_bytree in param_grid["colsample_bytree"]:
                            for gamma in param_grid["gamma"]:
                                model = xgb.XGBRegressor(
                                    n_estimators=1000,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    min_child_weight=min_child_weight,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,
                                    gamma=gamma,
                                    random_state=self.random_state,
                                    tree_method=(
                                        "gpu_hist" if self.use_gpu else "auto"
                                    ),
                                    early_stopping_rounds=50,
                                )

                                model.fit(
                                    self.X_train,
                                    self.y_train,
                                    eval_set=[(self.X_val, self.y_val)],
                                    verbose=False,
                                )

                                y_pred = model.predict(self.X_val)
                                rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

                                if rmse < best_rmse:
                                    best_rmse = rmse
                                    best_params = {
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "min_child_weight": min_child_weight,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample_bytree,
                                        "gamma": gamma,
                                    }
                                    best_model = model

        self.models["xgboost"] = best_model
        self.best_params["xgboost"] = best_params
        self._evaluate_model("xgboost")

        return self

    def train_lightgbm(self):
        """
        Train and optimize a LightGBM model using grid search.

        The method performs a grid search over specified hyperparameters and selects
        the best model based on validation RMSE. Early stopping is used to prevent
        overfitting.

        Returns
        -------
        self : object
            Returns the instance itself
        """
        print("\nTraining LightGBM...")

        param_grid = {
            "n_estimators": [1000],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50, 70],
            "min_child_samples": [20, 30, 50],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
        }

        best_rmse = float("inf")
        best_params = None
        best_model = None

        for max_depth in param_grid["max_depth"]:
            for learning_rate in param_grid["learning_rate"]:
                for num_leaves in param_grid["num_leaves"]:
                    for min_child_samples in param_grid["min_child_samples"]:
                        for subsample in param_grid["subsample"]:
                            for colsample_bytree in param_grid["colsample_bytree"]:
                                model = lgb.LGBMRegressor(
                                    n_estimators=1000,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    num_leaves=num_leaves,
                                    min_child_samples=min_child_samples,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,
                                    random_state=self.random_state,
                                    device=("gpu" if self.use_gpu else "cpu"),
                                )

                                model.fit(
                                    self.X_train,
                                    self.y_train,
                                    eval_set=[(self.X_val, self.y_val)],
                                    eval_metric="rmse",
                                    callbacks=[lgb.early_stopping(50, verbose=False)],
                                )

                                y_pred = model.predict(self.X_val)
                                rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

                                if rmse < best_rmse:
                                    best_rmse = rmse
                                    best_params = {
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "num_leaves": num_leaves,
                                        "min_child_samples": min_child_samples,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample_bytree,
                                    }
                                    best_model = model

        self.models["lightgbm"] = best_model
        self.best_params["lightgbm"] = best_params
        self._evaluate_model("lightgbm")

        return self

    def train_catboost(self):
        """
        Train and optimize a CatBoost model using grid search.

        The method performs a grid search over specified hyperparameters and selects
        the best model based on validation RMSE. Early stopping is used to prevent
        overfitting.

        Returns
        -------
        self : object
            Returns the instance itself
        """
        print("\nTraining CatBoost...")

        param_grid = {
            "iterations": [1000],
            "depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [3, 5, 7],
            "subsample": [0.8, 0.9],
        }

        best_rmse = float("inf")
        best_params = None
        best_model = None

        for depth in param_grid["depth"]:
            for learning_rate in param_grid["learning_rate"]:
                for l2_leaf_reg in param_grid["l2_leaf_reg"]:
                    for subsample in param_grid["subsample"]:
                        model = cb.CatBoostRegressor(
                            iterations=1000,
                            depth=depth,
                            learning_rate=learning_rate,
                            l2_leaf_reg=l2_leaf_reg,
                            subsample=subsample,
                            random_state=self.random_state,
                            task_type="GPU" if self.use_gpu else "CPU",
                            verbose=False,
                        )

                        model.fit(
                            self.X_train,
                            self.y_train,
                            eval_set=(self.X_val, self.y_val),
                            early_stopping_rounds=50,
                            verbose=False,
                        )

                        y_pred = model.predict(self.X_val)
                        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {
                                "depth": depth,
                                "learning_rate": learning_rate,
                                "l2_leaf_reg": l2_leaf_reg,
                                "subsample": subsample,
                            }
                            best_model = model

        self.models["catboost"] = best_model
        self.best_params["catboost"] = best_params
        self._evaluate_model("catboost")

        return self

    def _evaluate_model(self, model_name):
        """
        Evaluate a trained model on the test set.

        Calculates and prints RMSE, MAE, and R² scores for the specified model.
        Also stores the evaluation metrics in the scores dictionary.

        Parameters
        ----------
        model_name : str
            Name of the model to evaluate ('xgboost', 'lightgbm', or 'catboost')
        """
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)

        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        self.scores[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

        print(f"\nEvaluation results for {model_name.upper()} on test set:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Best parameters: {self.best_params[model_name]}")

    def plot_feature_importance(self):
        """
        Plot feature importance for all trained models.

        Creates a horizontal bar plot showing the relative importance of each feature
        for each trained model. The plots are arranged in a 1x3 grid.
        """
        n_features = len(self.X_train.columns)
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, (model_name, model) in enumerate(self.models.items()):
            if model_name == "xgboost":
                importance = model.feature_importances_
            elif model_name == "lightgbm":
                importance = model.feature_importances_
            else:  # catboost
                importance = model.feature_importances_

            feat_importance = pd.DataFrame(
                {"feature": self.X_train.columns, "importance": importance}
            )
            feat_importance = feat_importance.sort_values("importance", ascending=True)

            axes[idx].barh(range(n_features), feat_importance["importance"])
            axes[idx].set_yticks(range(n_features))
            axes[idx].set_yticklabels(feat_importance["feature"])
            axes[idx].set_title(f"{model_name.upper()} Feature Importance")

        plt.tight_layout()
        plt.show()

    def plot_predictions(self):
        """
        Plot actual vs predicted values for all trained models.

        Creates scatter plots comparing actual values with model predictions for each
        trained model. Also includes a perfect prediction line for reference.
        The plots are arranged in a 1x3 grid.
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for idx, (model_name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)

            axes[idx].scatter(self.y_test, y_pred, alpha=0.5)
            axes[idx].plot(
                [self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                "r--",
                lw=2,
            )
            axes[idx].set_xlabel("Actual Values")
            axes[idx].set_ylabel("Predicted Values")
            axes[idx].set_title(f"{model_name.upper()} Predictions vs Actual")

        plt.tight_layout()
        plt.show()

    def save_models(self, path="models/"):
        """
        Save trained models and their evaluation results using joblib.

        Saves all trained models, their best parameters, and evaluation scores
        to the specified directory using joblib.dump(). Files are named with
        timestamps for versioning. Joblib is used instead of pickle as it's more
        efficient for saving scikit-learn compatible models and large numpy arrays.

        Parameters
        ----------
        path : str, optional (default="models/")
            Directory path where models and results will be saved
        """
        import os

        if not os.path.exists(path):
            os.makedirs(path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models using joblib
        for model_name, model in self.models.items():
            model_path = os.path.join(path, f"{model_name}_{timestamp}.joblib")
            joblib.dump(model, model_path, compress=3)

        # Save parameters and results
        results = {"best_params": self.best_params, "scores": self.scores}
        results_path = os.path.join(path, f"results_{timestamp}.joblib")
        joblib.dump(results, results_path, compress=3)

        print(f"\nModels and results saved successfully at: {path}")

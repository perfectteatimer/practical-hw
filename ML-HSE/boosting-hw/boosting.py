from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from typing import Optional


def score(clf, x, y):
    return roc_auc_score(y, clf.predict_proba(x)[:, 1])  


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str = 'Bernoulli',
        goss: bool = False,
        goss_k: float = 0.2,
        rsm: float = 1.0,
        quantization_type: Optional[str] = None,
        nbins: int = 255,
        random_state: Optional[int] = None, 
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.goss = goss
        self.goss_k = goss_k
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.random_state = random_state  

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}
        self.current_predictions = None  

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)  # Исправьте формулу на правильную. 
    
    def _bootstrap(self, X, y):
            n_samples = X.shape[0]
            if self.bootstrap_type == 'Bernoulli':
                mask = np.random.rand(n_samples) < self.subsample
                return X[mask], y[mask]
            elif self.bootstrap_type == 'Bayesian':
                weights = (-np.log(np.random.uniform(size=n_samples))) ** self.bagging_temperature
                return X, y * weights
            else:
                raise ValueError("Invalid bootstrap_type. Choose anotHer one.")
    def _convert_labels(self, y):
        """ {0, 1} to {-1, 1}."""
        return 2 * y - 1
    
    def _goss_sample(self, X, grad):
        n_samples = X.shape[0]
        top_k = int(self.goss_k * n_samples)

        def select_indices():
            top_grad_indices = np.argsort(np.abs(grad))[-top_k:]
            remaining_indices = np.setdiff1d(np.arange(n_samples), top_grad_indices)
            subsample_size = int(self.subsample * len(remaining_indices))
            sampled_indices = np.random.choice(remaining_indices, size=subsample_size, replace=False)
            return np.concatenate([top_grad_indices, sampled_indices]), len(remaining_indices), subsample_size

        selected_indices, remaining_len, subsample_size = select_indices()
        weights = np.ones(len(selected_indices))
        weights[len(selected_indices) - subsample_size:] *= remaining_len / subsample_size
        return X[selected_indices], grad[selected_indices] * weights
    
    def _rsm_sample(self, X):
        n_features = X.shape[1]
        selected_features = np.random.choice(n_features, size=int(self.rsm * n_features), replace=False)
        return X[:, selected_features], selected_features

    def _quantize(self, X):
        bins = (
            np.linspace(np.min(X, axis=0), np.max(X, axis=0), self.nbins + 1)
            if self.quantization_type == 'Uniform'
            else np.percentile(X, q=np.linspace(0, 100, self.nbins + 1), axis=0)
        )
        return np.digitize(X, bins) - 1 if self.quantization_type else X
        
    def partial_fit(self, X, y):
        y = self._convert_labels(y) 
        if self.current_predictions is None:
            self.current_predictions = np.zeros(X.shape[0])

        grad = -self.loss_derivative(y, self.current_predictions)

        X_sampled, grad_sampled = self._bootstrap(X, grad)

        model = self.base_model_class(**self.base_model_params)
        model.fit(X_sampled, grad_sampled)
        predicts = model.predict(X)
        gamma = self.find_optimal_gamma(y, self.current_predictions, predicts)

        self.models.append(model)
        self.gammas.append(gamma)

        self.current_predictions += gamma * predicts

            
    def fit(self, X_train, y_train, eval_set=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        y_train = self._convert_labels(y_train) 
        if eval_set:
            X_val, y_val = eval_set[0]
        else:
            X_val, y_val = None, None

        if self.early_stopping_rounds:
            best_val_score = float('-inf')
            improvement_counter = 0

        for i in range(self.n_estimators):
            self.partial_fit(X_train, y_train)

            train_auc = score(self, X_train, y_train)
            self.history["train_roc_auc"].append(train_auc)
            self.history["train_loss"].append(self.loss_fn(y_train, self.current_predictions))

            if X_val is not None and y_val is not None:
                val_auc = score(self, X_val, y_val)
                val_loss = self.loss_fn(y_val, self.predict_proba(X_val)[:, 1])
                self.history["val_roc_auc"].append(val_auc)
                self.history["val_loss"].append(val_loss)

                if self.early_stopping_rounds and val_auc > best_val_score:
                    best_val_score = val_auc
                    self.best_iteration = i
                    improvement_counter = 0
                elif self.early_stopping_rounds:
                    improvement_counter += 1
                    if improvement_counter >= self.early_stopping_rounds:
                        print(f"Early STOP at iteration {i+1}")
                        break

        if plot:
            if X_val is not None and y_val is not None:
                self.plot_history(X_val, y_val)
            else:
                self.plot_history(X_train, y_train)

    def predict_proba(self, X):
        pred = sum(gamma * model.predict(X) for model, gamma in zip(self.models, self.gammas))
        positive = self.sigmoid(pred) 
        negative = 1 - positive   
        return np.column_stack((negative, positive))

    
    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)


    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        plt.plot(self.history['train_loss'], label="Train Loss")
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label="Validation Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Over Iterations")
        plt.show()

    @property
    def feature_importances_(self):
        if not self.models:
            raise ValueError("No models fitted yet")
        
        total_importances = sum(
            model.feature_importances_ for model in self.models
        )
        return total_importances / total_importances.sum()


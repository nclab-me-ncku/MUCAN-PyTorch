import numpy as np
from tqdm import tqdm
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from tools import get_system_info


class RidgeRegression(object):
    """
    Class for the Wiener Filter Decoder
    Using sklearn.linear_model.Ridge
    """
    def __init__(self):
        self.model = linear_model.Ridge()
    
    def fit(self, x: list[np.ndarray], y: list[np.ndarray], n_lags: int):
        """
        Train Wiener Filter Decoder
        """
        self.n_lags = n_lags
        x_arr, y_arr = self._format_data(x, y, n_lags)
        alphas = np.logspace(1, 5, 10)  # scale between 10 and 100,000
        
        self.alpha = self._parameter_fit_with_sweep(x_arr, y_arr, alphas)
        self.model.set_params(alpha=self.alpha)
        self.model.fit(x_arr, y_arr)
    
    def fit_with_kfold(self, x: list[np.ndarray], y: list[np.ndarray],
                       n_lags: int, n_splits=4, verbose=1) -> float:
        """
        Train Wiener Filter Decoder with k-fold cross validation
        """
        self.decoders = dict()
        kf = KFold(n_splits=n_splits, shuffle=False)
        with tqdm(total=n_splits, desc='Training') as pbar:
            for train_idx, test_idx in kf.split(x):
                
                x_train = [x[i] for i in train_idx]
                y_train = [y[i] for i in train_idx]
                self.fit(x_train, y_train, n_lags=n_lags)

                x_test = [x[i] for i in test_idx]
                y_test = [y[i] for i in test_idx]
                y_test_pred = self.predict(x_test)
                r2 = r2_score(
                    np.concatenate(y_test), np.concatenate(y_test_pred),
                    multioutput='variance_weighted'
                )
                self.decoders.update({r2: (self.model, train_idx)})
                pbar.set_postfix(get_system_info())
                pbar.update()
        self.train_idx = self.decoders[max(self.decoders.keys())][1]
        self.model = self.decoders[max(self.decoders.keys())][0]
        return max(self.decoders.keys())
    
    def predict(self, x: list[np.ndarray]) -> list[np.ndarray]:
        x_arr, _ = self._format_data(x, x, self.n_lags)
        y_pred = self.model.predict(x_arr)
        return self._split_to_trials(y_pred)
    
    def _format_data(self, x: list[np.ndarray], y: list[np.ndarray], n_lags: int):
        self.lengths = [len(item) for item in x]
        self.sections = [sum(self.lengths[:i]) for i, _ in enumerate(self.lengths, 1)][:-1]
        
        x_prime = np.concatenate([self._add_lags(item, n_lags) for item in x])
        y_prime = np.concatenate(y)
        return x_prime.reshape(len(x_prime), -1), y_prime
    
    def _add_lags(self, arr: np.ndarray, n_lags: int, zero_padding=True) -> np.ndarray:
        pad = np.pad(arr, [(n_lags - 1, 0), (0, 0)])
        n_stacks = range(n_lags - 1, -1, -1)
        arrs = np.vstack([np.roll(pad[np.newaxis], i, axis=1) for i in n_stacks])
        arrs = arrs.transpose(1, 0, 2)[n_lags-1:]
        return arrs if zero_padding else arrs[n_lags-1:]
    
    def _split_to_trials(self, arr: np.ndarray) -> list[np.ndarray]:
        lst = np.split(arr, self.sections)
        return lst
    
    def _parameter_fit_with_sweep(self, x: np.ndarray, y: np.ndarray, alphas, kf=KFold(n_splits=4)):
        """
        Alpha
        """
        self.alphas = dict()
        for alpha in alphas:
            results = list()
            self.model.set_params(alpha=alpha)
            for train_idx, test_idx in kf.split(x):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                self.model.fit(x_train, y_train)
                y_test_pred = self.model.predict(x_test)
                results.append(r2_score(y_test, y_test_pred, multioutput='variance_weighted'))
                
            self.alphas.update({np.mean(results): alpha})
        return self.alphas[max(self.alphas.keys())]


class LinearRegression(object):
    """
    Class for the Wiener Filter Decoder
    This simply leverages the scikit-learn linear regression.
    """
    def __init__(self):
        return

    def fit(self, x: list[np.ndarray], y: list[np.ndarray], n_lags: int):
        """
        Train Wiener Filter Decoder
        """
        self.n_lags = n_lags
        x_train, y_train = self._format_data(x, y, n_lags=self.n_lags)
        
        self.model = linear_model.LinearRegression()
        self.model.fit(x_train, y_train)
        
    def fit_with_kfold(self, x: list[np.ndarray], y: list[np.ndarray],
                       n_lags: int, n_splits=4, verbose=1):
        """
        Train Wiener Filter Decoder with k-fold cross validation
        """
        self.decoders = dict()
        kf = KFold(n_splits=n_splits, shuffle=False)
        with tqdm(total=n_splits, desc='Training') as pbar:
            for train_idx, test_idx in kf.split(x):
                
                x_train = [x[i] for i in train_idx]
                y_train = [y[i] for i in train_idx]
                self.fit(x_train, y_train, n_lags=n_lags)

                x_test = [x[i] for i in test_idx]
                y_test = [y[i] for i in test_idx]
                
                y_test_pred = self.predict(x_test)
                r2 = r2_score(
                    np.concatenate(y_test), np.concatenate(y_test_pred),
                    multioutput='variance_weighted'
                )
                self.decoders.update({r2: self.model})
                pbar.update()
                
        if verbose >= 2:    
            print(f'The multi-variate R\u00b2 are: {list(self.decoders.keys())}')
            
        self.model = self.decoders[max(self.decoders.keys())]
        return max(self.decoders.keys())  # TODO: no return?

    def predict(self, x: list[np.ndarray]) -> list[np.ndarray]:
        """
        Predict outcomes using trained Wiener Cascade Decoder
        """
        x_test, _ = self._format_data(x, x, n_lags=self.n_lags)
        y_pred = self.model.predict(x_test)
        
        return self._split_to_trials(y_pred)
    
    def _format_data(self, x: list[np.ndarray], y: list[np.ndarray], n_lags: int) -> np.ndarray:
        self.lengths = [len(item) for item in x]
        self.sections = [sum(self.lengths[:i]) for i, _ in enumerate(self.lengths, 1)][:-1]
        
        x_prime = np.concatenate([self._add_lags(item, n_lags) for item in x])
        y_prime = np.concatenate(y)
        return x_prime.reshape(len(x_prime), -1), y_prime
    
    def _add_lags(self, arr: np.ndarray, n_lags: int, zero_padding=True) -> np.ndarray:
        pad = np.pad(arr, [(n_lags - 1, 0), (0, 0)])
        n_stacks = range(n_lags - 1, -1, -1)
        arrs = np.vstack([np.roll(pad[np.newaxis], i, axis=1) for i in n_stacks])
        arrs = arrs.transpose(1, 0, 2)[n_lags-1:]
        return arrs if zero_padding else arrs[n_lags-1:]
    
    def _split_to_trials(self, arr: np.ndarray) -> list[np.ndarray]:
        lst = np.split(arr, self.sections)
        return lst
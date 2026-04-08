import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


import numpy as np

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    где $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
    а $H(R) = 1-p_1^2-p_0^2$, где $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних (при сортировке) значений признака.
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых значениях критерия Джини следует выбрать минимальный порог.
    * За наличие в функции циклов балл будет снижен. Нужно векторизовать.

    :param feature_vector: вещественнозначный вектор признака (numpy array)
    :param target_vector: вектор классов (0 или 1), len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами
    :return ginis: вектор со значениями критерия Джини для каждого порога
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)

    # есл все значения признака одинаковы то возвращаем пустые массивы
    if np.all(feature_vector == feature_vector[0]):
        return np.array([]), np.array([]), None, None

    sort_idx = np.argsort(feature_vector)
    X_sorted = feature_vector[sort_idx]
    y_sorted = target_vector[sort_idx]

    n = len(X_sorted)

    c1_cum = np.cumsum(y_sorted)
    c1_total = c1_cum[-1]
    c0_total = n - c1_total

    # находим индексы, где признак меняется
    distinct_mask = np.diff(X_sorted) > 0
    # индексы для разбиения: между элементами i и i+1 (т.е. split происходит после i+1-го элемента)
    split_positions = np.where(distinct_mask)[0] + 1

    # если нет ни одного разбиения
    if len(split_positions) == 0:
        return np.array([]), np.array([]), None, None

    # пороги - среднее между соседними различающимися значениями признака (как и в прошлых посылках)
    thresholds = (X_sorted[split_positions - 1] + X_sorted[split_positions]) / 2

    #  размеры левого и правого подмножеств для каждого потенциального сплита
    left_count = split_positions
    right_count = n - left_count

    # колво- объектов класса 1 слева
    c1_l = c1_cum[split_positions - 1]
    # колво объектов класса 0 слева
    c0_l = left_count - c1_l

    # колво объектов класса 1 справа
    c1_r = c1_total - c1_l
    # колво объектов класса 0 справа
    c0_r = c0_total - c0_l

    H_l = 1 - (c1_l**2 + c0_l**2) / (left_count**2)
    H_r = 1 - (c1_r**2 + c0_r**2) / (right_count**2)

    ginis = - (left_count / n) * H_l - (right_count / n) * H_r

    # оптимальный порог (максимум ginis)
    # при равенстве берется первый максимум что и есть мин порог т.к трешкхолды отсортированы
    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best




class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) == 1:
                continue

            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError("Unknown feature type")

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        left_indices = split
        right_indices = np.logical_not(split)

        if np.sum(left_indices) < self._min_samples_leaf or np.sum(right_indices) < self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[left_indices], sub_y[left_indices], node["left_child"], depth + 1)
        self._fit_node(sub_X[right_indices], sub_y[right_indices], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_index = node["feature_split"]
        feature_type = self._feature_types[feature_index]
        feature_value = x[feature_index]

        if feature_type == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


class LinearRegressionTree(DecisionTree):
    def __init__(self, feature_types, base_model_type=None,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 n_quantiles=10, loss_function='mse'):
        if np.any([x != "real" and x != "categorical" for x in feature_types]):
            raise ValueError

        super().__init__(feature_types=feature_types,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf)
        
        self.n_quantiles = n_quantiles
        self.loss_function = loss_function
        self.base_model_type = base_model_type if base_model_type is not None else LinearRegression

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["model"] = self.base_model_type().fit(sub_X, sub_y)
            return

        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["model"] = self.base_model_type().fit(sub_X, sub_y)
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["model"] = self.base_model_type().fit(sub_X, sub_y)
            return

        feature_best, threshold_best, loss_best, split = None, None, float('inf'), None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            feature_vector = sub_X[:, feature]

            if feature_type == "real":
                thresholds = self._get_thresholds(feature_vector)
                for threshold in thresholds:
                    left_indices = feature_vector < threshold
                    right_indices = ~left_indices

                    if np.sum(left_indices) < self._min_samples_leaf or np.sum(right_indices) < self._min_samples_leaf:
                        continue

                    loss = self._compute_split_loss(sub_X, sub_y, left_indices, right_indices)
                    if loss is not None and loss < loss_best:
                        feature_best = feature
                        threshold_best = threshold
                        loss_best = loss
                        split = left_indices

            elif feature_type == "categorical":
                categories = feature_vector
                category_counts = Counter(categories)
                sorted_categories = [cat for cat, cnt in category_counts.most_common()]
                category_to_int = {cat: idx for idx, cat in enumerate(sorted_categories)}
                feature_vector_transformed = np.array([category_to_int[cat] for cat in categories])

                thresholds = self._get_thresholds(feature_vector_transformed)
                for threshold in thresholds:
                    left_indices = feature_vector_transformed < threshold
                    right_indices = ~left_indices

                    if np.sum(left_indices) < self._min_samples_leaf or np.sum(right_indices) < self._min_samples_leaf:
                        continue

                    loss = self._compute_split_loss(sub_X, sub_y, left_indices, right_indices)
                    if loss is not None and loss < loss_best:
                        feature_best = feature
                        threshold_best = threshold
                        loss_best = loss
                        split = left_indices
                        node['category_to_int'] = category_to_int
            else:
                raise ValueError("Unknown feature type")

        if feature_best is None:
            node["type"] = "terminal"
            node["model"] = self.base_model_type().fit(sub_X, sub_y)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _compute_split_loss(self, X, y, left_indices, right_indices):
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        if len(y_left) == 0 or len(y_right) == 0:
            return None

        model_left = self.base_model_type()
        model_left.fit(X_left, y_left)

        model_right = self.base_model_type()
        model_right.fit(X_right, y_right)

        y_pred_left = model_left.predict(X_left)
        y_pred_right = model_right.predict(X_right)

        if self.loss_function == 'mse':
            loss_left = mean_squared_error(y_left, y_pred_left)
            loss_right = mean_squared_error(y_right, y_pred_right)
        elif self.loss_function == 'mae':
            loss_left = mean_absolute_error(y_left, y_pred_left)
            loss_right = mean_absolute_error(y_right, y_pred_right)
        else:
            raise ValueError("Unknown loss function")

        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        total_loss = (n_left / n_total) * loss_left + (n_right / n_total) * loss_right
        return total_loss

    def _get_thresholds(self, feature_values):
        quantiles = np.linspace(0, 1, self.n_quantiles + 2)[1:-1]
        thresholds = np.quantile(feature_values, quantiles)
        thresholds = np.unique(thresholds)
        return thresholds

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict([x])[0]

        feature_index = node["feature_split"]
        threshold = node["threshold"]
        feature_type = self._feature_types[feature_index]

        feature_value = x[feature_index]

        if feature_type == 'real':
            if feature_value < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == 'categorical':
            category_to_int = node.get('category_to_int', {})
            category = feature_value
            feature_value_transformed = category_to_int.get(category, -1)
            if feature_value_transformed < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])
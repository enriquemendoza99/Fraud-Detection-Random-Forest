# Imports essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency
from pandas.api.types import is_numeric_dtype, is_categorical_dtype, is_bool_dtype, is_object_dtype
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
np.random.seed(12)


# Reads training data and prints first 5 lines
train_data_full = pd.read_csv("data/train.csv")
print(train_data_full.head())

# Split data in features and classifications
y = train_data_full["isFraud"]
x = train_data_full.drop(["isFraud", "TransactionID"], axis=1)
print(y.shape)
print(x.shape)

# Check total missing values
print("NaN values:", x.isna().sum().sum())
print("'NotFound' values:", (x == "NotFound").sum().sum())

# Convert "NotFound" value to NaN
x = x.replace("NotFound", np.nan)

# Seperates Numerical and Categorical Data
cat_cols = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"]
num_cols = [c for c in x.columns if c not in cat_cols]

print("Categorical columns:", cat_cols)
print("Number of categorical columns:", len(cat_cols))
print("First few numeric columns:", num_cols)
print("Number of numeric columns:", len(num_cols))

# Fill missing categorical values with most common value in that column
for c in cat_cols:
    mode = x[c].mode(dropna=True)
    if not mode.empty:
        x[c] = x[c].fillna(mode.iloc[0])
    else:
        x[c] = x[c].fillna("__NA__")

# Fill missing numerical values with the median of the values in that column
for c in num_cols:
    median = x[c].median()
    x[c] = x[c].fillna(median)

# Ensure all missing values have been filled
print("Remaining missing values:", x.isna().sum().sum())

# Splits the data intro training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    x, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# FOR DEBUGGING - Ensures Categorical Data is seen as Categorical
X_train[cat_cols] = X_train[cat_cols].astype('category')
X_val[cat_cols] = X_val[cat_cols].astype('category')

# Validate sizes of training and validation sets
print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)

# Validate even split of data
print("Fraud rate (full): ", np.mean(y))
print("Fraud rate (train):", np.mean(y_train))
print("Fraud rate (val):  ", np.mean(y_val))

# Calculate imbalance ratio and WTotal penalty
Y_count = y_train.value_counts()
imbalance_ratio = (Y_count.min() / Y_count.max()) * 3
print("Imbalance ratio: ", imbalance_ratio)
WTotal = (Y_count.max()*imbalance_ratio) + Y_count.min()
print("WTotal: ", WTotal)

# Calculate probability of both classifications with WTotal penalty to use at each node
def calc_prob_with_penalty(curr_y_train, WTotal, imbalance_ratio):
    Y_count = curr_y_train.value_counts()
    prob_notFraud = (Y_count.max()*imbalance_ratio) / WTotal
    prob_Fraud = Y_count.min() / WTotal
    return prob_notFraud, prob_Fraud 

# Check initial probabilities 
prob_notFraud, prob_Fraud = calc_prob_with_penalty(y_train, WTotal, imbalance_ratio)
print("Probability of not Fraud: ", prob_notFraud)
print("Probability of Fraud: ", prob_Fraud)

# Universal Calculation Functions
def calc_imbalance_ratio(y_train):
    # Counts the number of times each class/label appears in y_train
    y_count = y_train.value_counts()
    # Checks to see if only one class/label is present
    if len(y_count) < 2:
        return 1.0
    majority_class = y_count.max()
    minority_class = y_count.min()
    return float(majority_class) / float(minority_class)

def calculate_weights(y_train):
    # Counts the number of times each class/label appears in y_train
    y_count = y_train.value_counts()

    total_length = len(y_train)
    unique_classes = y_count.shape[0]

    weights = {}
    for class_id, count in y_count.items():
        weights[class_id] = total_length / (unique_classes * count)

    return weights


def calc_weighted_probabilites(y_train, weights=None):
    # Counts the number of times each class/label appears in y_train
    y_count = y_train.value_counts()

    # Determines if the node is empty
    if len(y_count) == 0:
        return np.array([])
    
    # Determines if weights have been provided
    # If no weights provided, calculates basic probability
    if weights is None:
        probability = y_count.values / y_count.values.sum()
        return probability
    
    # Calculates weighted probability
    # Gets weight from weights passed in, then it calculates the weighted value, then normalizes the value
    weight = np.array([weights.get(class_id, 1.0) for class_id in y_count.index])
    weighted_value = y_count.values * weight
    probability = weighted_value / weighted_value.sum()

    return probability


def simple_entropy(y_train, weights=None):
    probabilities = calc_weighted_probabilites(y_train, weights)

    # Checks to see if the node is pure 
    if probabilities.size == 0:
        return 0.0
    
    probabilities = probabilities[probabilities > 0]

    return -np.sum(probabilities * np.log2(probabilities))


def simple_gini(y_train, weights=None):
    probabilities = calc_weighted_probabilites(y_train, weights)
    
    # Checks to see if the node is pure 
    if probabilities.size == 0:
        return 0.0
    
    return 1.0 - np.sum(probabilities ** 2)


def simple_misclas(y_train, weights=None):
    if len(y_train) == 0:
        return 0.0
    
    probabilities = calc_weighted_probabilites(y_train, weights)
    max_probability = probabilities.max()

    # Checks to see if the node is pure 
    if probabilities.size == 0:
        return 0.0

    return 1.0 - max_probability


def calc_info_gain_simple(y_train, left_y, right_y, impurity = "entropy", weights=None):
    # Determines which method will be used to calculate impurity
    if impurity == "entropy":
        randomness = simple_entropy
    elif impurity == "gini":
        randomness = simple_gini
    else:
        randomness = simple_misclas
    
    # Gets the length of all data frames passed into the function
    total_length = len(y_train)
    left_length = len(left_y)
    right_length = len(right_y)

    # Ensures that both branches have data
    if left_length == 0 or right_length == 0:
        return 0.0
    
    total_entropy = randomness(y_train, weights)
    weighted_entropy = (left_length/total_length) * randomness(left_y, weights) + (right_length/total_length) * randomness(right_y, weights)

    return total_entropy - weighted_entropy


def threshold_calculation(feature, X_train, y_train, impurity="entropy", use_weights=True):

    x = X_train[feature]
    y = y_train

    weights = calculate_weights(y_train) if use_weights else None

    # Sorts data to make sure best threshold can be found between two adjacent values
    order = np.argsort(x.values)
    x_sorted = x.values[order]
    y_sorted = y.values[order]

    thresholds = []
    for i in range(1, len(x_sorted)):
        # Ensures that the feature value is not the same as the feature adjacent to it
        if x_sorted[i] != x_sorted[i-1] and y_sorted[i] != y_sorted[i-1]:
            # Calculates midpoint between feature values to create a potential threshold
            potential_threshold = (x_sorted[i] + x_sorted[i-1]) / 2.0
            thresholds.append(potential_threshold)

    # Initialize best info gain and best threshold
    best_info_gain = -np.inf
    best_threshold = None

    # Calculates info gain of each possible threshold to determine which is best
    for threshold in thresholds:
        less_than_threshold = (x <= threshold)
        greater_than_threshold = ~less_than_threshold
        info_gain = calc_info_gain_simple(y, y[less_than_threshold], y[greater_than_threshold], impurity = impurity, weights = weights)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_threshold = threshold

    return best_info_gain, best_threshold



class Node:
    def __init__(self, default, splitfeature=None, value=None, rule=None, children={}, depth=0):
        self.default = default
        self.splitfeature = splitfeature
        self.value = value
        self.rule = rule
        self.children = children
        self.depth = depth

class DecisionTree:
    def __init__(self, splitcriterion='entropy', alpha=0.05, maximumdepth=10, numerical=None, categorical=None):
        self.splitcriterion = splitcriterion
        self.alpha = alpha
        self.maximumdepth = maximumdepth
        self.targetclass = [0, 1]
        self.treeroot = None
        self.numerical = numerical
        self.categorical = categorical
        self.default = None


    def bestAttribute(self, X_train, y_train, imbalance_ratio):
        selected_feature = None
        rule = None
        max_info_gain = -np.inf
        features = X_train.columns

        for feature in features:
            if feature in self.numerical:
                weights = calculate_weights(y_train)
                info_gain, threshold = threshold_calculation(feature, X_train, y_train, self.splitcriterion, weights) 
                
                if info_gain > max_info_gain:
                    selected_feature = feature
                    rule = ("Threshold", threshold)
                    max_info_gain = info_gain

            else:
                categories = X_train[feature].unique()
                for category in categories:
                    x = X_train[feature]
                    category_filter = (x == category)
                    left_y = y_train[category_filter]
                    right_y = y_train[~category_filter]
                    weights = calculate_weights(y_train)
                    info_gain = calc_info_gain_simple(y_train, left_y, right_y, self.splitcriterion, weights)
                
                    if info_gain > max_info_gain:
                        selected_feature = feature
                        rule = ("Category", category)
                        max_info_gain = info_gain

        return selected_feature, rule, max_info_gain


    def chisquare(self, X_train, y_train, feature):
        if len(y_train.unique()) == 1:
            return True

        contingency = pd.crosstab(X_train[feature], y_train)
        chisquare, pvalue, df, expected = chi2_contingency(contingency)

        # If the pvalue is greater than the alpha that means that the features are independent
        return pvalue > self.alpha

    def train(self, X_train, y_train, imbalance_ratio):
        self.default = np.bincount(y_train).argmax()
        self.treeroot = self.expandTree(X_train, y_train, imbalance_ratio, 0)


    def expandTree(self, X_train, y_train, imbalance_ratio, depth):
        selected_feature, rule, max_info_gain = self.bestAttribute(X_train, y_train, imbalance_ratio)
        children = {}

        # Chi-Squared Stopping Criteria
        if self.chisquare(X_train, y_train, selected_feature):
            return Node(default=self.default, value=np.bincount(y_train).argmax())
        # Creates leaf node (only contains a value)
        if depth >= self.maximumdepth or len(y_train.value_counts()) == 1:
            return Node(default=self.default, value = np.bincount(y_train).argmax())
        
        # Creates child sub-trees
        if selected_feature in self.numerical:
            less_than_threshold = X_train[selected_feature] <= rule[1]
            greater_than_threshold = ~less_than_threshold

            X_left, y_left =  X_train[less_than_threshold], y_train[less_than_threshold]
            X_right, y_right = X_train[greater_than_threshold], y_train[greater_than_threshold]

            left_node = self.expandTree(X_left, y_left, imbalance_ratio, depth + 1)
            right_node = self.expandTree(X_right, y_right,imbalance_ratio, depth + 1)
            children = {"left":left_node, "right":right_node}
        else:
            x = X_train[selected_feature]
            x_unique_categories = list(set(x))

            for category in x_unique_categories:
                category_filter = (x == category)
                X_child, y_child = X_train[category_filter], y_train[category_filter]
                children[category] = self.expandTree(X_child, y_child, imbalance_ratio, depth + 1)
            

        return Node(default=self.default, splitfeature=selected_feature, value=None, rule=rule, children=children, depth=depth)
    

    def _the_judgement_(self, treeroot, x_row):
        if treeroot.value != None:
            return treeroot.value
        
        method, val = treeroot.rule

        if method == "Threshold":
            if x_row[treeroot.splitfeature] <= val:
                return self._the_judgement_(treeroot.children["left"], x_row)
            else:
                return self._the_judgement_(treeroot.children["right"], x_row)
        
        else:
            category = x_row[treeroot.splitfeature]
            if category in treeroot.children:
                return self._the_judgement_(treeroot.children[category], x_row)
            else:
                return self.treeroot.default


    def predict(self, X):
        return [self._the_judgement_(self.treeroot, row) for _, row in X.iterrows()]
    

            
class RandomForest:
    def __init__(self, num_trees=10, maximumdepth=10, splitcriterion = 'entropy', alpha = 0.05, numerical=None, categorical=None):
        self.trees = []
        self.num_trees = num_trees
        self.maximumdepth = maximumdepth
        self.splitcriterion = splitcriterion
        self.alpha = alpha
        self.numerical = numerical
        self.categorical = categorical

    def train(self, X_train, y_train, imbalance_ratio):
        for tree_num in range(self.num_trees):
            samples = len(X_train)
            randomindices = []
            for _ in range(samples):
                row = np.random.randint(0, samples)
                randomindices.append(row)

            Xarray = X_train.values
            yarray = y_train.values

            Xsampled = Xarray[randomindices]
            ysampled = yarray[randomindices]

            Xsample = pd.DataFrame(Xsampled, columns=X_train.columns)
            ysample = pd.Series(ysampled, name=y_train.name)

            tree = DecisionTree(
                splitcriterion=self.splitcriterion,
                alpha=self.alpha,
                maximumdepth=self.maximumdepth,
                numerical=self.numerical,
                categorical=self.categorical
            )
            print(f"\nTraining Tree Number {tree_num}")
            tree.train(Xsample, ysample, imbalance_ratio)
            print(f"Completed Training Tree Number {tree_num}")
            self.trees.append(tree)

    def predict(self, X_val):
        all_predictions = np.array([tree.predict(X_val) for tree in self.trees])
        all_predictions = np.swapaxes(all_predictions, 0, 1)
        prediction = np.array([np.bincount(row).argmax() for row in all_predictions])
        return prediction

# Train and Evaluate
print("\n\nTraining Initial Decision Tree...")
dt = DecisionTree(maximumdepth=15, numerical=num_cols, categorical=cat_cols)
dt.train(X_train, y_train, imbalance_ratio)
print("Decision Tree Training Complete\n")
print("Predicting from Initial Decision Tree...")
y_prediction = dt.predict(X_val)
print("\nDecision Tree Prediction Complete\n")

print("\n\nTraining Random Forest...")
rf = RandomForest(num_trees=20, maximumdepth=15, numerical=num_cols, categorical=cat_cols)
rf.train(X_train, y_train, imbalance_ratio)
print("Random Forest Training Complete")
print("\nPredicting from Random Forest...")
y_pred_rf = rf.predict(X_val)
print("\nRandom Forest Predictions Complete")

# Evaluation
print("\n\nEvaluating Decision Tree...\n")
print("Decision Tree Evaluation:")
dtaccuracy = accuracy_score(y_val, y_prediction)
dtbalancedaccuracy = balanced_accuracy_score(y_val, y_prediction)
dtconfusionmatrix = confusion_matrix(y_val, y_prediction)

print(f"Accuracy: {dtaccuracy:.4f}")
print(f"Balanced Accuracy: {dtbalancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{dtconfusionmatrix}")


print("\n\nEvaluating Random Forest...\n")
print("Random Forest Accuracy:")
rfaccuracy = accuracy_score(y_val, y_pred_rf)
rfbalancedaccuracy = balanced_accuracy_score(y_val, y_pred_rf)
rfconfusionmatrix = confusion_matrix(y_val, y_pred_rf)

print(f"Accuracy: {rfaccuracy:.4f}")
print(f"Balanced Accuracy: {rfbalancedaccuracy:.4f}")
print(f"Confusion Matrix:\n{rfconfusionmatrix}")


# TODO Uncomment when ready to test on Kaggle Data
# TODO Creates CSV for Kaggle Predictions
kaggle_test_data = pd.read_csv("data/test.csv")
kaggle_test_pred = rf.predict(kaggle_test_data)
TransactionID = kaggle_test_data['TransactionID']
kaggle_df = pd.DataFrame({'TransactionID': TransactionID, 'isFraud': kaggle_test_pred})
kaggle_df.to_csv("TheOutlierDetectivesPredictions.csv", index=False)

# PLOT 1 — Class Distribution (Fraud vs. Legitimate)

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor('#0b0f1a')
ax.set_facecolor('#111520')

counts = y_train.value_counts().sort_index()
labels = ['Legitimate', 'Fraudulent']
colors = ['#5ee7b0', '#ff6b6b']
bars = ax.bar(labels, counts.values, color=colors, width=0.45,
              edgecolor='none', zorder=3)

# Value labels on bars
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.values.max() * 0.015,
            f'{val:,}', ha='center', va='bottom',
            color='#c0c8e0', fontsize=11, fontweight='bold', fontfamily='monospace')

ax.set_title('Class Distribution — Fraud vs. Legitimate', color='#e8eaf0',
             fontsize=13, fontweight='bold', pad=14)
ax.set_ylabel('Number of Transactions', color='#5a6080', fontsize=10)
ax.tick_params(colors='#7a8099', labelsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.spines[:].set_visible(False)
ax.grid(axis='y', color='#1e2535', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('plot1_class_distribution.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: plot1_class_distribution.png")

# PLOT 2 — Feature Importance (Top 10 by Information Gain)

# Collect info gain per feature from the first tree in the forest
from collections import defaultdict


def collect_feature_gains(node, gains=None):
    """Recursively walk a tree and accumulate info gains per feature."""
    if gains is None:
        gains = defaultdict(float)
    if node is None or node.value is not None:
        return gains
    if node.splitfeature is not None:
        gains[node.splitfeature] += 1  # count splits as proxy for importance
    for child in node.children.values():
        collect_feature_gains(child, gains)
    return gains


all_gains = defaultdict(float)
for tree in rf.trees:
    tree_gains = collect_feature_gains(tree.treeroot)
    for feat, val in tree_gains.items():
        all_gains[feat] += val

# Normalize
total = sum(all_gains.values())
normalized = {k: v / total for k, v in all_gains.items()}

# Top 10
top10 = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:10]
feat_names, feat_vals = zip(*top10)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0b0f1a')
ax.set_facecolor('#111520')

colors_feat = ['#3b9eff' if i == 0 else '#3b9eff88' for i in range(len(feat_names))]
bars = ax.barh(feat_names[::-1], feat_vals[::-1], color=colors_feat[::-1],
               edgecolor='none', height=0.6, zorder=3)

ax.set_title('Top 10 Features by Split Frequency (Random Forest)',
             color='#e8eaf0', fontsize=13, fontweight='bold', pad=14)
ax.set_xlabel('Normalized Importance', color='#5a6080', fontsize=10)
ax.tick_params(colors='#7a8099', labelsize=10)
ax.spines[:].set_visible(False)
ax.grid(axis='x', color='#1e2535', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('plot2_feature_importance.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: plot2_feature_importance.png")

# PLOT 3 — Confusion Matrix (Random Forest)

cm = confusion_matrix(y_val, y_pred_rf)
acc = accuracy_score(y_val, y_pred_rf)
bal_acc = balanced_accuracy_score(y_val, y_pred_rf)

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('#0b0f1a')
ax.set_facecolor('#111520')

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Legitimate', 'Fraud'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')

# Style the matrix
ax.set_title(
    f'Confusion Matrix — Random Forest\nAccuracy: {acc:.2%}  |  Balanced Accuracy: {bal_acc:.2%}',
    color='#e8eaf0', fontsize=12, fontweight='bold', pad=12
)
ax.tick_params(colors='#7a8099', labelsize=10)
ax.xaxis.label.set_color('#7a8099')
ax.yaxis.label.set_color('#7a8099')
ax.spines[:].set_color('#1e2535')

# Update text color inside matrix cells
for text in disp.text_.ravel():
    text.set_color('white')
    text.set_fontsize(14)
    text.set_fontweight('bold')

plt.tight_layout()
plt.savefig('plot3_confusion_matrix.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: plot3_confusion_matrix.png")

print("\nAll 3 plots saved in your project folder.")


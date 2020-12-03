import numpy as np


def label_code(row):
    if row['id'] == 'Oregon':
        return 'OR'
    elif row['id'] == 'California':
        return 'CA'
    elif row['id'] == 'Washington':
        return 'WA'
    elif row['id'] == 'Kentucky':
        return 'KY'
    elif row['id'] == 'Texas':
        return 'TX'
    elif row['id'] == 'New York':
        return 'NY'
    elif row['id'] == 'Florida':
        return 'FL'
    elif row['id'] == 'Illinois':
        return 'IL'
    elif row['id'] == 'South Carolina':
        return 'SC'
    elif row['id'] == 'North Carolina':
        return 'NC'
    elif row['id'] == 'Georgia':
        return 'GA'
    elif row['id'] == 'Virginia':
        return 'VA'
    elif row['id'] == 'Ohio':
        return 'OH'
    elif row['id'] == 'Wyoming':
        return 'WY'
    elif row['id'] == 'Missouri':
        return 'MO'
    elif row['id'] == 'Montana':
        return 'MT'
    elif row['id'] == 'Utah':
        return 'UT'
    elif row['id'] == 'Minnesota':
        return 'MN'
    elif row['id'] == 'Mississippi':
        return 'MS'
    elif row['id'] == 'Arizona':
        return 'AZ'
    elif row['id'] == 'Alabama':
        return 'AL'
    else:
        return 'MA'


def label_state(country):
    if country == 'France':
        return "Select Department"
    elif country == 'Australia':
        return "Select Region"
    elif country == 'Canada':
        return "Select Province"
    else:
        return "Select State"


def model_information(classification):
    if classification == 'Logistic Regression':
        text_1 = 'Logistic Regression is a Machine Learning classification algorithm that is used ' \
                 'to predict the probability of a categorical dependent variable. '
        text_2 = 'Only the meaningful variables should be included. ' \
                 'The independent variables should be independent of each other. ' \
                 'That is, the model should have little or no multicollinearity. '
        text_3 = 'The independent variables are linearly related to the log odds. ' \
                 'Logistic regression requires quite large sample sizes.'
    elif classification == 'Support Vector Machine':
        text_1 = 'The objective of the support vector machine algorithm is to find a hyperplane in an ' \
                 'N-dimensional space(where N is the number of features) that distinctly classifies the data points.'
        text_2 = 'To separate the two classes of data points, there are many possible hyperplanes that ' \
                 'could be chosen. ' \
                 'Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. ' \
                 'Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.'
        text_3 = 'Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. ' \
                 'Using these support vectors, we maximize the margin of the classifier.'
    elif classification == 'K-Nearest Neighbors':
        text_1 = 'The KNN algorithm assumes that similar things exist in close proximity. ' \
                 'KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) ' \
                 'by calculating the distance between points on a graph. The straight-line distance (also called the Euclidean distance) is a popular and familiar choice.'
        text_2 = 'Disadvantages : The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase.'
        text_3 = 'The algorithm is simple and easy to implement. There’s no need to build a model, tune several parameters, or make additional assumptions.' \
                 'The algorithm is versatile. It can be used for classification, regression, and search (as we will see in the next section).'
    elif classification == 'Naive Bayes':
        text_1 = 'A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. ' \
                 'The crux of the classifier is based on the Bayes theorem.'
        text_2 = 'Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. ' \
                 'That is presence of one particular feature does not affect the other. Hence it is called naive.'
        text_3 = 'Gaussian Naive Bayes:When the predictors take up ' \
                 'a continuous value and are not discrete, we assume that these values are sampled ' \
                 'from a Gaussian or Normal distribution. Naive Bayes algorithms are fast and easy to implement but ' \
                 'their biggest disadvantage is that the requirement of predictors to be independent. In most of the real life cases, the predictors are dependent, this hinders the performance of the classifier.'
    elif classification == 'Decision Tree':
        text_1 = 'Decision tree learning is a supervised machine learning technique for inducing a decision tree from training data. A decision tree (also referred to as a classification tree or a reduction tree) is a predictive model' \
                 ' which is a mapping from observations about an item to conclusions about its target value'
        text_2 = 'In the tree structures, leaves represent classifications (also referred to as labels), ' \
                 'nonleaf nodes are features, and branches represent conjunctions of features that lead to the classifications.' \
                 'Building a decision tree that is consistent with a given data set is easy. The challenge lies in building good decision trees, which typically means the smallest decision trees. '
        text_3 = 'Advantages: ' \
                 'Easy to interpret and explain to nontechnical users.' \
                 'Decision trees require relatively little effort from users for data preparation. ' \
                 'Disadvantage: without proper pruning or limiting tree growth, they tend to overfit the training data, making them somewhat poor predictors.'
    else:
        text_1 = 'Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest ' \
                 'spits out a class prediction and the class with the most votes becomes our model’s prediction'
        text_2 = 'A large number of relatively uncorrelated models (trees) operating as a ' \
                 'committee will outperform any of the individual constituent models.'
        text_3 = 'Random forest ensure that the behavior of each individual tree is not ' \
                 'too correlated with the behavior of any of the other trees in the model using ' \
                 'Bagging (Bootstrap Aggregation) and Feature Randomness. '

    return text_1, text_2, text_3


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def select_classification(classification):
    if classification == "Logistic Regression":
        classifier = LogisticRegression()
        tuned_parameters = {
            'C': np.linspace(.0001, 1000, 200),
            'penalty': ['l2']
        }
    elif classification == "Support Vector Machine":
        classifier = SVC(probability=True)
        tuned_parameters = {
            'kernel': ['rbf'],
            'gamma': ['auto', 'scale'],
            'degree': [3, 8],
            'C': [1, 10, 100]
        }
    elif classification == "K-Nearest Neighbors":
        classifier = KNeighborsClassifier()
        tuned_parameters = {
            'leaf_size': list(range(1, 30)),
            'n_neighbors': list(range(1, 10)),
            'p': [1, 2]
        }
    elif classification == "Naive Bayes":
        classifier = GaussianNB()
        tuned_parameters = {
            'var_smoothing': np.logspace(0, -9, num=100)
        }
    elif classification == "Decision Tree":
        classifier = DecisionTreeClassifier()
        tuned_parameters = {
            'max_depth': np.linspace(1, 32, 32, endpoint=True),
            'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
            'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True)
            #'max_features': list(range(1, X_train.shape[1]))
        }
    else:
        classifier = RandomForestClassifier()
        tuned_parameters = {
            'min_samples_split': [3, 5, 10],
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 15, 25]
            # 'max_features': list(range(1, X_train.shape[1]))
        }
    return classifier, tuned_parameters

from sklearn import tree
from sklearn import ensemble
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC




models = {
    'decision_tree_gini': tree.DecisionTreeClassifier(
        criterion='gini'
    ),
    'decision_tree_entropy': tree.DecisionTreeClassifier(
        criterion='entropy'
    ),
    'extra_rf': ensemble.ExtraTreesClassifier(random_state = 42),
    'rf': ensemble.RandomForestClassifier(),
    'xgb': XGBClassifier(),
    'svc': SVC(),
    'k_nearest': KNeighborsClassifier(),
    'log_reg': LogisticRegression(),

}

## list of commands:
#  python train.py --fold 0 --model rf
#  python train.py --fold 0 --model decision_tree_gini
#  python train.py --fold 0 --model decision_tree_entropy
#  python train.py --fold 0 --model xgb
#  python train.py --fold 0 --model svc
#  python train.py --fold 0 --model k_nearest

#rf and xgboost are the best atm.
# use: bash run_tests.sh
#then put the no folds and the model in.
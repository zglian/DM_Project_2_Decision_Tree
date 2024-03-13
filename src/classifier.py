import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import pydotplus
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

GENDER = 'gender'
EXHAUST_VOLUME = 'exhaust_volume'
MOTORCYCLE_STYLE = 'motorcycle_style'
PASSENGER_ON_BACKSEAT = 'passenger_on_backseat'
HELMET = 'helmet'
WEIGHT = 'weight'
HEIGHT = 'height'
AGE = 'age'
PURCHASE_PRICE = 'purchase_price'
MONTHLY_MILEAGE = 'monthly_mileage'
TRAFFIC_TICKET_COUNT = 'traffic_ticket_count'
SPEEDING_HABBIT = 'speeding_habbit'
HAS_LICENSE = 'has_license'
HAS_MODIFICATIONS = 'has_modifications'
IS_NOISY_ENGINE = 'is_noisy_engine'
IS_MOUNTAIN_RIDE = 'is_mountain_ride'
HAS_REGULAR_MAINTENANCE = 'has_regular_maintenance'
COMMUTE_WITH_MOTORCYCLE = 'commute_with_motorcycle'
LABEL = 'class'
HIGH_RISK = 'high_risk'
MEDIUM_RISK = 'medium_risk'
LOW_RISK = 'low_risk'
WEIGHT = 'weight'
HEIGHT = 'height'

features = [
    # GENDER,
    # EXHAUST_VOLUME,
    # MOTORCYCLE_STYLE,
    # PASSENGER_ON_BACKSEAT,
    # HELMET,
    WEIGHT,
    HEIGHT,
    AGE,
    PURCHASE_PRICE,
    MONTHLY_MILEAGE,
    TRAFFIC_TICKET_COUNT,
    SPEEDING_HABBIT,
    HAS_LICENSE,
    HAS_MODIFICATIONS,
    IS_NOISY_ENGINE,
    IS_MOUNTAIN_RIDE,
    HAS_REGULAR_MAINTENANCE,
    COMMUTE_WITH_MOTORCYCLE,
]

def decision_tree():
    print("========= Decision Tree =========")
    print(f"train data: {dataset1_filename}")
    print(f"test data: {dataset2_filename}")
    
    # create dicison tree classifier
    clf = DecisionTreeClassifier(min_samples_leaf=30, max_depth=6)
    # 在訓練集上擬合模型
    clf.fit(X_train, y_train)
    # 用測試集評估模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy with Decidion Tree: {:.2f}%".format(accuracy * 100))

    #generate decision tree plot
    dot_data = export_graphviz(clf, out_file=None,
                    feature_names=features,
                    class_names=[HIGH_RISK, MEDIUM_RISK, LOW_RISK],
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(f"./output/decision_tree.png")

    #generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    # plt.show()
    plt.savefig(f"./output/decision_tree_cm.png")

    print("Success.")


def naive_bayes():
    print("========= Naive Bayes =========")
    print(f"train data: {dataset1_filename}")
    print(f"test data: {dataset2_filename}")

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred_gnb = clf.predict(X_test)
    accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
    print("Accuracy with Gaussian Naive Bayes: {:.2f}%".format(accuracy_gnb * 100))

    #generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_gnb, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    
    disp.plot()
    # plt.show()
    plt.savefig(f"./output/naive_bayes_cm.png")

    print("Success.")

def svm():
    print("============ SVM =============")
    print(f"train data: {dataset1_filename}")
    print(f"test data: {dataset2_filename}")

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("Accuracy on test data with SVM: {:.2f}%".format(accuracy_svm * 100))

    #generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_svm, labels=svm.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
    disp.plot()
    # plt.show()
    plt.savefig(f"./output/svm_cm.png")

    print("Success.")

def knn():
    print("============ KNN ===========")
    print(f"train data: {dataset1_filename}")
    print(f"test data: {dataset2_filename}")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("Accuracy on test data with KNN: {:.2f}%".format(accuracy_knn * 100))

    #generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_knn, labels=knn.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
    disp.plot()
    # plt.show()
    plt.savefig(f"./output/knn_cm.png")

    print("Success.")


dataset1_filename = "dataset1-7000.csv"
dataset2_filename = "dataset2-3000.csv"
train_data = pd.read_csv(f"inputs/{dataset1_filename}")
test_data = pd.read_csv(f"inputs/{dataset2_filename}")
#convert string to interger
train_data = pd.get_dummies(train_data, columns=[GENDER, EXHAUST_VOLUME, MOTORCYCLE_STYLE, PASSENGER_ON_BACKSEAT, HELMET])
test_data = pd.get_dummies(test_data, columns=[GENDER, EXHAUST_VOLUME, MOTORCYCLE_STYLE, PASSENGER_ON_BACKSEAT, HELMET])

X_train = train_data[features]
y_train = train_data[LABEL]
X_test = test_data[features]
y_test = test_data[LABEL]

if __name__ == '__main__':
   decision_tree()
   naive_bayes()
   knn()
   svm()


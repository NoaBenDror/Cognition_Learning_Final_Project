############################################################################
#            6119 COMPUTATION AND COGNITION PROJECT
#                   Noa Ben-Dror 316163260
#                   Michal Dagan 315657064
############################################################################

############################################################################
#            packages imports
############################################################################

import plotly.express as px
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


############################################################################
#            code
############################################################################

# build the 3-D graph
def plot_in_plotly(res, method, labels, s=6, addon="", discrete=True, save=False):
    """
        plots data in 3D
        res : 3D array
        method : method used for graph
        labels : labels for data (could just pass though np.zeros(res.shape[0])
        s : size of point
        addon : additional info for title of graph
        discrete : whether the labels data is discrete or continuous
    """
    df = pd.DataFrame(res, columns=[method + '_1', method + '_2', method + '_3'])
    df['labels'] = labels
    fig2 = px.scatter_3d(df, x=method + '_1', y=method + '_2', z=method + '_3', color='labels')
    fig2.update_layout(
        title=method + " Plot " + addon,
        template="plotly_dark"
    )
    fig2.update_traces(marker=dict(size=s))
    if save:
        fig2.write_html("" + method + " Plot " + addon + ".html")
    else:
        fig2.show()


def pca():
    """
    This function runs the PCA algorithm
    """

    # read the data
    data = pd.read_csv('C:\\Users\\206mi\\Desktop\\pcomputer\\data.csv', header=0)
    target = data["risk"].values

    plt.figure(figsize=(20, 10))

    sns.heatmap(data.drop(['risk'], axis=1).corr(), annot=True)
    plt.title("Correlation Plot of Mean Features")
    plt.show()

    data = data.drop(columns="risk")
    data_scaled = StandardScaler().fit_transform(data)
    # perform PCA, d=3
    data_PCA_3 = PCA(n_components=3).fit_transform(data_scaled)
    plot_in_plotly(data_PCA_3, "PCA", target, addon="3D data", save=True, s=8)

    # perform PCA, d=12
    data_PCA_12 = PCA(n_components=12).fit(data_scaled)
    print(data_PCA_12.explained_variance_)
    print(data_PCA_12.explained_variance_ratio_)
    print(data_PCA_12.components_)

    plt.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], data_PCA_12.explained_variance_ratio_,
            width=0.5,
            tick_label=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9', 'pc10',
                        'pc11', 'pc12'])
    plt.xlabel('PC')
    plt.ylabel('Explained Variance')
    plt.show()

    # perform PCA, d=2
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data_scaled)
    print("comp", pca.components_)
    print("var", pca.explained_variance_)

    print('Explained variation per principal component: {}'.format(
        pca.explained_variance_ratio_))

    # save the first two principal components into csv and then add them the risk column
    # df = pd.DataFrame(list(zip(*[principalComponents[:, 0], principalComponents[:, 1]]))).add_prefix('PC')
    # df.to_csv('file.csv', index=False)

    plt.plot(np.cumsum(PCA().fit(data).explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    plt.plot(range(0, 2), pca.explained_variance_ratio_)
    plt.ylabel('Explained Variance')
    plt.xlabel('Principal Components')

    plt.title('Explained Variance Ratio')
    plt.show()

    exp_var_pca = pca.explained_variance_ratio_

    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    from bioinfokit.visuz import cluster

    pca_out = PCA().fit(data_scaled)
    loadings = pca_out.components_
    pca_scores = PCA().fit_transform(data_scaled)

    cluster.biplot(cscore=pca_scores, loadings=loadings, labels=data.columns.values,
                   var1=round(pca_out.explained_variance_ratio_[0] * 100, 2),
                   var2=round(pca_out.explained_variance_ratio_[1] * 100, 2), show=True,
                   colorlist=target)


pca()


class Perceptron:
    """Perceptron classifier"""

    def __init__(self):
        self.w = None
        self.b = 0  # the bias

    def predict_one(self, x):
        """ Calculate the dot product of the features and the weights.
        :param x - sample
        :return the label 0 or 1
        """
        return 1 if (np.dot(self.w, x) >= self.b) else 0

    def predict(self, X):
        """ receives the samples and creates a list for each x of the predictions
        :param X - samples
        :return np array of the predictions for X
        """
        Y = []
        for x in X:
            result = self.predict_one(x)
            Y.append(result)
        return np.array(Y)

    def getwb(self):
        """
        :return: returns the bias and the weight
        """
        return self.w, self.b

    def fit(self, X, Y, X_test, Y_test, epochs=10000, lr=0.0001):
        """
        Receives samples and labels and trains the computer to find the optimal
        weights and bias. (that gives Minimum errors)
        :param X: Training dataset
        :param y: Binary classification of dataset.
        :param epochs: the number of epochs
        :param lr: learning rate
        """
        self.b = 0
        self.w = np.ones(X.shape[1])
        accuracy = {}  # saves the accuracy of each update
        accuracy_test = {}
        max_accuracy = 0
        idx_max = 0
        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.predict_one(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1
            y_pred = self.predict(X)
            n_errors = sum(y_pred != Y)
            if n_errors == 0:  # if find perfect separation
                break
            accuracy[i] = accuracy_score(Y, y_pred)
            accuracy_test[i] = accuracy_score(Y_test, self.predict(X_test))  # without test erase

            if accuracy[i] > max_accuracy:
                max_accuracy = accuracy[i]
                idx_max = i
        plt.plot(list(accuracy.values()), label="train_acc")
        plt.plot(list(accuracy_test.values()), label="test_acc")  # without test erase
        return max_accuracy, idx_max

#######################
# runs the perceptron:
#######################

plt.style.use("seaborn")
data = pd.read_csv('C:\\Users\\206mi\\Desktop\\pcomputer\\data.csv', header=0)
X = data.drop('risk', axis=1)
Y = data['risk']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20
                                                    , stratify=Y,
                                                    random_state=1)  # 80% training and 20% test

perceptron = Perceptron()

max_accuracy, idx_max = perceptron.fit(np.array(X_train), np.array(y_train), np.array(X_test),
                                       np.array(y_test), 20000, 0.00001)
Y_pred_test = perceptron.predict(np.array(X_test))

print(accuracy_score(Y_pred_test, y_test))
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title(
    "Accuracy of Perceptron model with  $\eta$ = 0.00001 ")
plt.ylim([0, 1])
plt.legend()
plt.show()
#######################

def SVM():
    """
    This function runs the SVM algorithm
    """
    df = pd.read_csv('C:\\Users\\206mi\\Desktop\\pcomputer\\data.csv', header=0)
    y = df['risk']
    X = df.drop(['risk'], axis=1)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=Y,
                                                        random_state=0)  # 80% training and 20% test

    print("train:", X_train.shape, y_train.shape)
    print("test:", X_test.shape, y_test.shape)

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    # comparing actual test set values and predicted values.
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))

    print(metrics.classification_report(y_test, y_pred=y_pred))
    # get support vectors
    # print("Support vectors:", clf.support_vectors_)

    b_svm = clf.intercept_[0]
    w_svm = clf.coef_[0]
    print("SVM w: {} b: {}".format(w_svm, b_svm))

    plot_confusion_matrix(clf, X_test, y_test,
                          cmap=plt.cm.GnBu,
                          normalize='true')
    plt.title('Normalized Confusion matrix')
    plt.show()

    plot_confusion_matrix(clf, X_test, y_test,
                          cmap=plt.cm.GnBu,
                          normalize=None)
    plt.title('Confusion matrix')
    plt.show()

    """ 
    Support Vector Classifier
    There are several kernels for the Support Vector Classifier.
    we will test some of them and check which has the best score."""
    svc_scores = []
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in range(len(kernels)):
        svc_classifier = svm.SVC(kernel=kernels[i])
        svc_classifier.fit(X_train, y_train)
        svc_scores.append(svc_classifier.score(X_test, y_test))

    colors = plt.cm.Blues(np.linspace(0, 1, len(kernels)))
    plt.bar(kernels, svc_scores, color=colors)
    for i in range(len(kernels)):
        plt.text(i, svc_scores[i], svc_scores[i])
    plt.xlabel('Kernels')
    plt.ylabel('Scores')
    plt.title('Support Vector Classifier scores for different kernels')
    plt.show()

    df = pd.read_csv('C:\\Users\\206mi\\Desktop\\pcomputer\\file.csv', header=0)
    # show principal Components 1 and 2 with svm and Perceptron
    p1, p2 = 'PC1', 'PC2'
    X = df[[p1, p2]].to_numpy()
    show(X, y, p1, p2, df)


def show(X, y, p1, p2, df):
    """
    Creates perceptron and svm. And for each of them finds the weights and bias.
    From them we will create the dividing line.
    :param X: samples
    :param y: labels
    :param p1, p2: the labels names
    :param df: for the hue the color by the label
    """
    plt.style.use("seaborn")

    perc = Perceptron()  # use Perceptron fit function without the test data
    perc.fit(X, y, 20000, 0.01)

    svm_c = svm.SVC(kernel='linear')
    svm_c.fit(X, y)
    b_svm = svm_c.intercept_[0]
    w_svm = svm_c.coef_[0]
    fig, ax = plt.subplots(figsize=(8, 8))

    plt.ylim([-5, 5])
    s = sns.scatterplot(X[:, 0], X[:, 1], hue=df['risk'], legend='brief',
                        palette=["lawngreen", "deeppink"], s=60)
    s.set_xlabel(p1, fontsize=20)
    s.set_ylabel(p2, fontsize=20)
    s.set_title("principal Components 1 and 2 with svm and Perceptron", fontsize=20)

    # creating the line:
    dots = [np.amin(X[:, 0]), np.max(X[:, 0])]
    w, b = perc.getwb()

    sns.lineplot(x=dots, y=[-(x * w[0] + b) / w[1] for x in dots], color='blue', label='precpetron')
    sns.lineplot(x=dots, y=[-(x * w_svm[0] + b_svm) / w_svm[1] for x in dots], color='cyan',
                 label='svm')

    print("Number of support vectors for each class.", svm_c.n_support_)
    support_vectors = svm_c.support_vectors_
    # Highlight support vectors with a circle around them
    ax.scatter(svm_c.support_vectors_[:, 0], svm_c.support_vectors_[:, 1], s=100, linewidth=1,
               facecolors='none', edgecolors='k')

    plt.show()


SVM()


def build_model():
    """ small network to model the binary classification problem."""
    # Building model
    a = Input(12)  # Input dimensions
    x = Dense(8, activation='relu')(a)
    final_result = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=a, outputs=final_result)
    return model


def deep_neural_network():
    """
    This function runs deep neural network
    """
    df = pd.read_csv('C:\\Users\\206mi\\Desktop\\pcomputer\\data.csv', header=0)
    y = df['risk']
    X = df.drop(['risk'], axis=1)

    model = build_model()
    # compiling model
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy', metrics=['accuracy'])
    # training model with 15 epochs
    # history = model.fit(X, y, epochs=100)
    # training model with validation set in size 20% with 100 epochs
    history = model.fit(X, y, epochs=100, validation_split=0.20)
    print(history.history.keys())

    plt.figure()
    plt.plot(history.history['loss'], label="train_loss")
    plt.plot(history.history['accuracy'], label="train_acc")
    plt.plot(history.history['val_loss'], label="val_loss")
    plt.plot(history.history['val_accuracy'], label="val_acc")
    plt.title("Training Loss and Accuracy [Epoch {}]".format(100))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


deep_neural_network()

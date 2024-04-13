from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_confusion_matrix(classifier, features, target, labels):
    """Plots the confusion matrix for a given classifier."""
    disp = ConfusionMatrixDisplay(classifier, features, target, include_values=True, display_labels=labels)
    disp.plot()
    plt.show()


def main():
    # Load iris dataset
    data = load_iris()
    features = data.data
    target = data.target

    # Original feature space visualization
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(features[:, 0], features[:, 1], c=target, marker='o', cmap='viridis')
    plt.title("Original Feature Space")

    # MLP Classifier with original features
    classifier = MLPClassifier(hidden_layer_sizes=(10), alpha=1, max_iter=100)
    classifier.fit(features, target)
    prediction = classifier.predict(features)

    plt.subplot(2, 2, 3)
    plt.scatter(features[:, 0], features[:, 1], c=prediction, marker='d', cmap='viridis', s=150)
    plt.scatter(features[:, 0], features[:, 1], c=target, marker='o', cmap='viridis', s=15)
    plt.title("MLP Classifier Prediction (Original Features)")

    # PCA dimensionality reduction
    pca = PCA(n_components=2, whiten=True, svd_solver='randomized')
    pca.fit(features)
    pca_features = pca.transform(features)
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f'Preserved Variance: {explained_variance:.2f}% of the original data information')

    plt.subplot(2, 2, 2)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=target, marker='o', cmap='viridis')
    plt.title("PCA Transformed Features")

    # MLP Classifier with PCA features
    classifier_pca = MLPClassifier(hidden_layer_sizes=(600, 300, 150, 75, 20), alpha=1, max_iter=40000)
    classifier_pca.fit(pca_features, target)
    prediction_pca = classifier_pca.predict(pca_features)

    plt.subplot(2, 2, 4)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=prediction_pca, marker='d', cmap='viridis', s=150)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=target, marker='o', cmap='viridis', s=15)
    plt.title("MLP Classifier Prediction (PCA Features)")

    plt.tight_layout()
    plt.show()

    # Confusion Matrices
    plot_confusion_matrix(classifier, features, target, data.target_names)
    plot_confusion_matrix(classifier_pca, pca_features, target, data.target_names)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from data_preprocessing import load_and_preprocess

# Load dataset
df = load_and_preprocess("data/spam_mail_data.csv")

# Load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")

X = vectorizer.transform(df['cleaned_text']).toarray()
y = df['label']

# Reduce dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Train SVM in 2D (for visualization only)
model = SVC(kernel="linear")
model.fit(X_reduced, y)

# Plot points
plt.figure(figsize=(8,6))

plt.scatter(
    X_reduced[:,0],
    X_reduced[:,1],
    c=y,
    cmap="coolwarm",
    alpha=0.5
)

# Plot hyperplane
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(
    XX,
    YY,
    Z,
    colors='black',
    levels=[0]
)

plt.title("SVM Spam vs Ham Decision Boundary (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.savefig("app/static/svm_plot.png")
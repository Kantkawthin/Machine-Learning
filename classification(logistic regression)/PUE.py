'''
Predicting University Enrollment Using Logistic Regression

'''

#import libiray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset using numpy (Replace load_data with np.loadtxt)
data = np.loadtxt("ex2data1.txt", delimiter=",")  # Load dataset
X = data[:, :-1]  # Features (Exam scores)
y = data[:, -1]   # Labels (Admission: 0 or 1)

#view the data
print(f'The first 5 row of X_train:\n {X[:5]}')
print(f'The first 5 row of Y_train:\n {y[:5]}')


#view number of training example
print(f'The shape of X_train is: {X.shape}')
print(f'The shape of y_train is: {y.shape}')
print(f'We have m = {len(y)} training examples')



# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Standardize the features  (important for logistic regression)
#Standardize means changing numbers so they are on the same scale.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Plot the data with decision boundary
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="b", marker="o", label="Admitted")
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="r", marker="x", label="Not Admitted")


# Create a grid of values for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))



# Predict for each grid point
Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors="g")

# Set labels and legend
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.7)
plt.title("Logistic Regression Decision Boundary")

# Show the plot
plt.show()

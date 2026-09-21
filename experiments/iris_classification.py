"""Train the three one-vs-rest logistic classifiers from the Iris experiment."""

from __future__ import annotations

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_foundations.supervised.classification import OneVsRestLogistic
from ml_foundations.supervised.preprocessing import standardize


iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
x_train, x_test, _, _ = standardize(x_train, x_test)
y_train = torch.tensor(y_train, dtype=torch.long)

torch.manual_seed(0)
model = OneVsRestLogistic(num_features=4, num_classes=3)
model.fit(x_train, y_train, learning_rate=0.1, iterations=1000)
predictions = model.predict(x_test).numpy()
accuracy = (predictions == y_test).mean()

print(f"Correct: {(predictions == y_test).sum()}/{len(y_test)}")
print(f"Accuracy: {accuracy:.2%}")

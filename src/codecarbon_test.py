from codecarbon import EmissionsTracker
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

tracker = EmissionsTracker()
tracker.start()

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

emissions = tracker.stop()

print("Accuracy:", accuracy)
print("CO2 Emissions:", emissions)
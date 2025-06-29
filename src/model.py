from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model(df):
    X = df.drop(columns='target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    return acc

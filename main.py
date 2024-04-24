import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dataAAPG = pd.read_csv(r"data/edad-altura-peso-gen.csv")
dataGAP = pd.read_csv(r"data/genero-altura-peso.csv")

train, test = train_test_split(dataGAP, test_size=0.2)

# print (train.describe())
# print (test.describe())

# prepare data
label = "gender"

y_train = train[label].values
y_test = test[label].values

X_train = train.drop([label], axis=1).values
X_test = test.drop([label], axis=1).values

# start training
reg_log = LogisticRegression()
reg_log.fit(X_train, y_train)

print(str(reg_log.predict([[50, 200], [40, 70]])))

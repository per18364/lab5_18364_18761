from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from analisis import *
from graphs import *
from SVM import SVM
# Carga el dataset de iris
df, df_temp = Data_clean()

df = Analisis_exploratorio(df)


X_train, X_test, y_train, y_test = train_test_split(
    df.drop('status', axis=1), df_temp['status'], test_size=0.2, random_state=1234)


clf = SVC(kernel='linear')

scores = cross_val_score(clf, X_train, y_train, cv=5)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Precisión de la libreria:', accuracy)

X_train = X_train.values
Y_train = y_train.values


svm = SVM()

svm.fit(X_train, Y_train)

Y_pred2 = svm.predict(X_test)

accuracy = accuracy_score(y_test, Y_pred2)
print('\nPrecisión nuestra:', accuracy)


graph3(df)
graph4(df)

similitud = 0
predictions = list(y_pred)
Y_pred2 = list(Y_pred2)
for i in range(len(y_pred)):
    similitud += Y_pred2[i] == predictions[i]

print("Cuanto se parecen ambos modelos: ", similitud/len(Y_pred2))

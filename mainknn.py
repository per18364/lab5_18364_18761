from analisis import *
from graphs import *

df, df_temp = Data_clean()
df = Analisis_exploratorio(df)
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('status', axis=1), df_temp['status'], test_size=0.2, random_state=1234)

# Predicción con libreria
knn = KNeighborsClassifier(n_neighbors=int(df.shape[0] ** 0.5))
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print('Precisión de la libreria:',
      accuracy_score(y_test, predictions))

# Calcular la matriz de confusión
print('\nMatriz de confusión:\n', confusion_matrix(y_test, predictions))

knn2 = KNN(n_neighbors=int(df.shape[0] ** 0.5))
knn2.fit(X_train, y_train)
predictions2 = knn2.predict(X_test)
print('\nPrecisión nuestra:',
      knn2.accuracy_score(y_test, predictions2))


graph1(df)
graph2(df)

similitud = 0
predictions = list(predictions)
for i in range(len(predictions2)):
    similitud += predictions2[i] == predictions[i]

print("Que tanto se parecen ambos modelos: ", similitud/len(predictions2))

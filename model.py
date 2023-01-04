import pandas as pd
from sklearn import tree
import pickle
df=pd.read_csv('py.csv')
input=df.drop(columns=['churn'])
target=df['churn']
model=tree.DecisionTreeClassifier()
model.fit(input,target)


pickle.dump(model, open('model.pkl','wb'))
model1 = pickle.load(open('model.pkl','rb'))



# result=model.predict([[45,60000,0,0]])
# print(result)

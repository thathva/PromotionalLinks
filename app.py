

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#finding the dataset
df1=pd.read_csv("Youtube01-Psy.csv")
df1.head()

df2=pd.read_csv("Youtube02-KatyPerry.csv")
df3=pd.read_csv("Youtube03-LMFAO.csv")
df4=pd.read_csv("Youtube04-Eminem.csv")
df5=pd.read_csv("Youtube05-Shakira.csv")

frames=[df1,df2,df3,df4,df5]
df_merged=pd.concat(frames)

df_merged

keys=["Psy","KatyPerry","LMFAO","Eminem","Shakira"]
df=pd.concat(frames,keys=keys)

df.to_csv("youtubecommentsdataset.csv")

df.isnull().isnull().sum()    
df.columns


df_data=df[['CONTENT','CLASS']]


df_x=df_data['CONTENT']
df_y=df_data['CLASS']

#countvectorizer
cv=CountVectorizer()
demo=cv.fit_transform(["trying this data","trying a new data"])
demo.toarray()
demo
cv.get_feature_names()


cv=CountVectorizer()
x=cv.fit_transform(df_x)

x.toarray()
cv.get_feature_names()

#training and testing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix


x.shape
x_train,x_test,y_train,y_test=train_test_split(x,df_y,test_size=0.33,random_state=42)

#naive bayes
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB()
clf.fit(x_train,y_train)
res=clf.score(x_test,y_test)
nbpred=clf.predict(x_test)

print("percentage of accurate result is ",res*100,"%")


print(accuracy_score(y_test,nbpred))
print(roc_auc_score(y_test,nbpred))
print(confusion_matrix(y_test,nbpred))

#logistic regression
from sklearn.linear_model import LogisticRegression

linclf = LogisticRegression(solver='liblinear', penalty='l1')
linclf.fit(x_train,y_train)
lrpred = linclf.predict(x_test)
acc=accuracy_score(y_test,lrpred)


print(accuracy_score(y_test,lrpred))
print(roc_auc_score(y_test,lrpred))
print(confusion_matrix(y_test,lrpred))


#random forest 

from sklearn.ensemble import RandomForestClassifier

#clf1=RandomForestClassifier(criterion='entropy',min_samples_split=4,max_features='log2')
clf1=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features=8, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf1.fit(x_train,y_train)
rfpred=clf1.predict(x_test)


print(accuracy_score(y_test,rfpred))
print(roc_auc_score(y_test,rfpred))
print(confusion_matrix(y_test,rfpred))

#just random data testing
comment=["check this link for additional reading"]
vect=cv.transform(comment).toarray()

if linclf.predict(vect)==1:
    print("spam")
else:
    print("not spam")

len(y_test)

x_test[0]

#finding no of correct msgs which were blocked
i=0
blockham=0
while (i < len(y_test)):
    if clf1.predict(x_test[i])[0] == 1 and y_test[i] == 0:
        #print(i)
        print(x_test[i])
        blockham=blockham+1        
    i=i+1    

blockham



from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    comment = request.form['comment']
    comment=[comment]
    val=cv.transform(comment)
    prediction=clf1.predict(val)
    if(prediction==0):
        #pred="Spam"
        msg = "The comment is not spam"
    else:
        msg = "This comment is spam"

    return render_template('result.html',pred=msg)

if __name__ == '__main__':
    
    app.run(port=5000,debug=True, use_reloader=False)

    















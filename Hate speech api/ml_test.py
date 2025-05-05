import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer 


#cv=CountVectorizer()
with open("model/Hs_model.pkl", "rb") as file:
    model =pk.load(file)
with open("model/Hs_cv.pkl", "rb") as file:
    cv =pk.load(file)
        
t=cv.transform(['nigga','bitch']).toarray()
print(model.predict(t))
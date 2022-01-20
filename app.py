from flask import Flask, render_template,request
import numpy as np

import pickle
from sklearn.feature_extraction.text import CountVectorizer
app=Flask(__name__)
model=pickle.load(open('Naivemodel.pkl','rb'))
modelVectorizer=pickle.load(open('cv.pickle','rb'))

@app.route('/')

def hello_world():
    
    return render_template("index2.html")


# Prediction
#def predict_nationality(x):
#	vect = modelVectorizer.transform(data).toarray()
#	result = nationality_clf.predict(vect)
#	return result
 
@app.route('/')
def index():
	return render_template('index2.html')


@app.route('/predection', methods=['POST'])

def predection():
    
   # sentenceInput = request.form.get("text")
   # input=[sentenceInput]
    #cv=CountVectorizer()

    #myvect=cv.transform([sentenceInput]).toarray()
   
    #prediction=model.predict(myvect)
    #pred_proba=model.predict_proba(myvect)
    #pred_percentage_for_all=dict(zip(model.classes_,pred_proba))
    
   # final_features = [np.array(sentenceInput)]
   # features = [int(x) for x in request.form.values()]

    text = request.form['text']
    data=[text]
    vect = modelVectorizer.transform(data).toarray()
    prediction = model.predict(vect)
    pred_proba=model.predict_proba(vect)
    pred_percentage_for_all=dict(zip(model.classes_,pred_proba))
    print("Prediction:{},Prediction Score{}".format(prediction[0],np.max(pred_proba)))
    score=np.max(pred_proba)*100
    #final_features = [np.array(Total_stops)]
   # prediction = model.predict(final_features)
    #output=round(prediction[0],1)
    return render_template('index2.html', prediction_text=prediction,name = text.upper(),prediScore=score)

   

if __name__ == '__main__':
    app.run(port=3000,debug=True)

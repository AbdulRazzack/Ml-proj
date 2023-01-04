from flask import Flask,render_template,request, url_for,jsonify
import numpy as np
import pickle


app=Flask(__name__)
model1 = pickle.load(open('model.pkl','rb'))


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST','GET'])
def predict():
    data =[[int(x) for x in request.form.values()]]
    data1=np.array(data)

    prediction = model1.predict(data1)

    if prediction==1:
        prediction_text="Customer will Churn!!"
    else:
        prediction_text="Customer will not Churn!!"

    return render_template("index.html", churn_text= prediction_text)







if __name__=="__main__":
    app.run(debug=True, port=2400)
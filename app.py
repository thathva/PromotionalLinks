from flask import Flask,request,render_template
#from flask_restful import Api
import joblib

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    comment = request.form['comment']
    x=count.transform([comment])
    prediction=clf1.predict(x)
    if(prediction==0):
        s="Not Spam"
    else:
        s="Spam"
    return render_template('result.html',pred=s)
if __name__ == '__main__':
    clf1=joblib.load('model.pkl')
    count=joblib.load('count.pkl')
    app.run(port=5000, debug=True)  # important to mention debug=True

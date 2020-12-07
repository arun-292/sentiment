from flask import Flask, render_template,redirect,request,url_for,session,jsonify
from Project import test_sample
from Project import logreg
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        senti=request.form['sentence']
        flag=test_sample(logreg, senti)
        return render_template('index.html',flag=flag,senti=senti)
    else:
        return render_template('index.html',flag=-1)
    
if __name__ == "__main__":
    app.secret_key='arun'
    app.run(debug=True)
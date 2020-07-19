from inference import get_result
from flask import Flask,render_template,request,jsonify
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html",text=None)

@app.route("/extract/",methods=["GET","POST"])
def extract_sentiment():
    content = request.get_json()
    text=content["text"]
    sentiment=content["sentiment"]
    extract=get_result(text,sentiment)
    # print(text,sentiment)
    # extract="dummy text just for texting"
    
    return jsonify({"extract":extract})
    # return render_template("index.html",text=extract)

if __name__=="__main__":
    app.run(port=3000,debug=True)
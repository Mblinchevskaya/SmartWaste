import numpy as np
import pandas as pd
import pickle
from flask import Flask, request
import json
import requests
from PIL import Image


app = Flask(__name__)


@app.route('/predictions_dict', methods=['GET'])
def predict_waste():
    url = request.args.get('url')
    # url = 'https://www.ayaakov.co.il/files/products/product1596_image1_2022-01-03_11-07-20.jpg'
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.resize((80, 80))
    pix = np.array(img)
    one_test = pd.DataFrame(pix.flatten()).T
    proba = loaded_model.predict_proba(one_test) #.tolist()[0]
    keys = [0, 1, 2, 3, 4]
    json_proba = json.dumps(dict(zip(keys, proba[0])))
    return json_proba


if __name__ == '__main__':
    loaded_model = pickle.load(open('churn_model.pkl', 'rb'))
    app.run(host='0.0.0.0', port=8080)

    # prediction = predict_waste()
    # print(prediction)



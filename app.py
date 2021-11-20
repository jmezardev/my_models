from flask import Flask
from flask import request
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('model.joblib')
labels = ['setosa', 'versicolor', 'virginica']

@app.route("/")
def home():
	return """
	<html>
	<h1>Modelo Iris</h1>


	<form action='predict' method='GET'>
	<label for="v1">v1</label>
	<input type="text" id="v1" name="v1"><br>
	<label for="v2">v2</label>
	<input type="text" id="v2" name="v2"><br>
	<label for="v3">v3</label>
	<input type="text" id="v3" name="v3"><br>
	<label for="v4">v4</label>
	<input type="text" id="v4" name="v4"><br>

	<input type="submit">
	</form>

	</html>

	"""

@app.route("/predict")
def predict():
	v1 = float(request.args.get('v1'))
	v2 = float(request.args.get('v2'))
	v3 = float(request.args.get('v3'))
	v4 = float(request.args.get('v4'))

	result = model.predict(np.array([[v1,v2,v3,v4]]))

	return "<h1> Prediction: {}</h1>".format(labels[result[0]])



if __name__ == '__main__':
	app.run(debug=False, use_reloader=True)
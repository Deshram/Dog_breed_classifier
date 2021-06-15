
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from flask import Flask, request, render_template

# Define a flask app
app = Flask(__name__)

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    
    x = preprocess_input(x)
    model = load_model('model.h5')    #loading resnet50 model
    preds = model.predict(x)
    return preds


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    f = request.files['file']
    preds = model_predict(f.filename)

    pred_class = decode_predictions(preds, top=1)   
    breed = str(pred_class[0][0][1])               
    score = str(pred_class[0][0][2])
    return {'breed':breed,'score':score}
    


if __name__ == '__main__':
    app.run(debug=True)

from fastapi import FastAPI, File, UploadFile # type: ignore
import uvicorn # type: ignore
import numpy as np # type: ignore
from io import BytesIO
from PIL import Image # type: ignore
import tensorflow as tf # type: ignore
import keras #type: ignore

app = FastAPI()

# MODEL = tf.keras.models.load_model(r"C:\Users\techno\Desktop\Driver Drowsiness Detection\Saved_Models\1")
# MODEL = tf.keras.models.Sequential([
MODEL = tf.keras.layers.TFSMLayer(r"C:\Users\techno\Desktop\Driver Drowsiness Detection\Saved_Models\1", call_endpoint="serving_default")
# ])
# MODEL = tf.saved_model.load(r'C:\Users\techno\Desktop\Driver Drowsiness Detection\Saved_Models\1')


CLASS_NAMES = ["Closed", "Open" , "no_yawn","yawn"]

@app.get("/ping")
async def ping():
    return "Hello , I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    prediction = MODEL(img_batch)
    one_dim_array = tf.squeeze(prediction['output_0'])
    predicted_class = CLASS_NAMES[np.argmax(one_dim_array)]
    
    confidence = np.max(one_dim_array)

    return {
        'class' : predicted_class,
        'confidence' : float(confidence)
    }
    


if __name__=="__main__":
    uvicorn.run(app, host='localhost',port=8000)
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Cargar el modelo Keras
#model = load_model("modelo_1_final.keras")
model = load_model("modelo_1_final2.h5")

#model.load_weights("modelo_1_peso_final.h5")
class_names = ['chavez', 'maduro', 'nofigura']  # reemplaza esto con tus nombres de clase reales

def load_and_prep_image(filename, img_shape=224):
    """
    Lee una imagen de un archivo, la convierte en un tensor y la remodela sin alterar su contenido.
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = img/255.
    return img

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("formulario.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile):
    # Guardar el archivo temporalmente
    with open("temp_image", "wb") as buffer:
        buffer.write(await file.read())

    # Preparar la imagen
    img = load_and_prep_image("temp_image", img_shape=224)

    # Hacer la predicción
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Añadir lógica para multi-clase y obtener el nombre de pred_class
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]

    # Devolver la clase predicha y las probabilidades de predicción
    # en una página HTML
    return f"""
    <html>
        <body>
            <h1 style="text-align:center;">Hola, el modelo de Cristian Cuevas te da el siguiente pronóstico:</h1>
            <h2 style="text-align:center;">Clase predicha: {pred_class}</h2>
            <h3 style="text-align:center;">Probabilidades de predicción: {pred[0].tolist()}</h3>
        </body>
    </html>
    """

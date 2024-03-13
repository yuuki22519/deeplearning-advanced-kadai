from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import os

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import save_model

model = VGG16(weights='imagenet')
save_model(model, 'vgg16.h5')

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
   
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = img_array / 255

            img_array = img_array[..., ::-1]  # RGB to BGR
            img_array = preprocess_input(img_array)

            result = model.predict(img_array)
            prediction = decode_predictions(result)[0]

            for class_name, description, probability in prediction:
                print(f"カテゴリ: {description}, 確率: {probability}")

            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': prediction, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})

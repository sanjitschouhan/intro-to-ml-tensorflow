import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from PIL import Image

import sys
import argparse
import json

image_size = 224

def load_model(model_file):
    return tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    
    processed_test_image = process_image(test_image)
    batched_test_image = np.expand_dims(processed_test_image, axis=0)
    
    preds = model.predict(batched_test_image)[0]
    ind = np.argsort(preds)[-top_k:][::-1]
    return preds[ind], ind+1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flower Image Classifier')
    parser.add_argument('image_path', metavar='image_path', type=str,
                    help='path to image file')
    parser.add_argument('model', metavar='model', type=str,
                    help='path to saved model file')
    parser.add_argument('--top_k', metavar='top_k', type=int, default=1,
                    help='returns top_k predictions')
    parser.add_argument('--category_names', metavar='category_names', type=str, default=None,
                    help='path to category names json file')

    args = parser.parse_args()
    
    model = load_model(args.model)
    
    class_names = dict()
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    
    probs, classes = predict(args.image_path, model, args.top_k)
    print()
    for cls, prob in zip(classes,probs):
        class_name = class_names.get(str(cls), cls)
        res = f"{class_name:20}: {prob*100:>7.3f}%"
        print(res)
    print()
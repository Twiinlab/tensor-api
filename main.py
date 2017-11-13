# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import logging


from flask import Flask, request, json, Response
from flask_restful import reqparse, abort, Api, Resource

# TensorFlow Dependencias

import os
import urllib
import tensorflow as tf
import numpy as np
from skimage import io
from skimage import transform as tform
from skimage.color import rgb2gray, rgb2grey, rgba2rgb

app = Flask(__name__)


from PIL import Image
import base64
import re
from io import BytesIO

api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')

TODOS = {
    'todo1': {'task': 'build an API'},
    'todo2': {'task': '?????'},
    'todo3': {'task': 'profit!'},
}


def abort_if_todo_doesnt_exist(todo_id):
    if todo_id not in TODOS:
        abort(404, message="Todo {} doesn't exist".format(todo_id))

parser = reqparse.RequestParser()
parser.add_argument('task')

# Todo
# shows a single todo item and lets you delete a todo item
class Todo(Resource):
    def get(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        return TODOS[todo_id]

    def delete(self, todo_id):
        abort_if_todo_doesnt_exist(todo_id)
        del TODOS[todo_id]
        return '', 204

    def put(self, todo_id):
        args = parser.parse_args()
        task = {'task': args['task']}
        TODOS[todo_id] = task
        return task, 201


# TodoList
# shows a list of all todos, and lets you POST to add new tasks
class TodoList(Resource):
    def get(self):
        return TODOS

    def post(self):
        args = parser.parse_args()
        todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
        todo_id = 'todo%i' % todo_id
        TODOS[todo_id] = {'task': args['task']}
        return TODOS[todo_id], 201

##
## Actually setup the Api resource routing here
##
api.add_resource(TodoList, '/todos')
api.add_resource(Todo, '/todos/<todo_id>')

# ## Init variables

savefile = "./STORED_model/my_trained_model.json"

# ### Helper Functions

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

def one_hot_encode(pos):
    out = np.zeros(10)
    out[pos] = 1
    return out

# # Placeholders
# x = tf.placeholder(tf.float32,shape=[None,784])
# y_true = tf.placeholder(tf.float32,shape=[None,10])


# # Layers
# x_image = tf.reshape(x,[-1,28,28,1])

# convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
# convo_1_pooling = max_pool_2by2(convo_1)

# convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
# convo_2_pooling = max_pool_2by2(convo_2)

# convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
# full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

# hold_prob = tf.placeholder(tf.float32)
# full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

# y_pred = normal_full_layer(full_one_dropout,10)

# # Loss Function
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

# Optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=0.0001) # 0.0001
# train = optimizer.minimize(cross_entropy)

# Initialize Variables
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()


# ### Session

imageURL = 'https://firebasestorage.googleapis.com/v0/b/tensorweb-af554.appspot.com/o/test01.png?alt=media&token=26c1db9f-b36b-420d-acce-48845753b9f9'
imageURL2 = 'https://firebasestorage.googleapis.com/v0/b/tensorweb-af554.appspot.com/o/testcar01.png?alt=media&token=a29a9ccd-1cd1-4731-9f17-7bab3a4bda98'
imageURL3 = 'https://firebasestorage.googleapis.com/v0/b/tensorweb-af554.appspot.com/o/test03.png?alt=media&token=d235eba9-d645-48f4-8616-825e19d35de6'


def predictFromDataImage(data):
    with tf.Session() as sess:
        # Placeholders
        x = tf.placeholder(tf.float32,shape=[None,784])
        y_true = tf.placeholder(tf.float32,shape=[None,10])


        # Layers
        x_image = tf.reshape(x,[-1,28,28,1])

        convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
        convo_1_pooling = max_pool_2by2(convo_1)

        convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
        convo_2_pooling = max_pool_2by2(convo_2)

        convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

        hold_prob = tf.placeholder(tf.float32)
        full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

        y_pred = normal_full_layer(full_one_dropout,10)

        # Loss Function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001) # 0.0001
        train = optimizer.minimize(cross_entropy)

        # restore the model
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # restore the model
        saver.restore(sess, "./STORED_model/my_trained_model.json")
        
        imageData = readGrayImageFromData(data)
        imageSimple = simplifyImage(imageData)
        
        feed_dict = {x: imageSimple, y_true: np.zeros((1, 10)), hold_prob : 0.5 }
        classification = sess.run(tf.argmax(y_pred,1), feed_dict)
        
    return classification

def predictFromUrlImage(imageUrl):
    with tf.Session() as sess:
        # Placeholders
        x = tf.placeholder(tf.float32,shape=[None,784])
        y_true = tf.placeholder(tf.float32,shape=[None,10])


        # Layers
        x_image = tf.reshape(x,[-1,28,28,1])

        convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
        convo_1_pooling = max_pool_2by2(convo_1)

        convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
        convo_2_pooling = max_pool_2by2(convo_2)

        convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

        hold_prob = tf.placeholder(tf.float32)
        full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

        y_pred = normal_full_layer(full_one_dropout,10)

        # Loss Function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001) # 0.0001
        train = optimizer.minimize(cross_entropy)

        # restore the model
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        saver.restore(sess, "./STORED_model/my_trained_model.json")
        
        imageData = readGrayImageFromUrl(imageUrl)
        imageSimple = simplifyImage(imageData)
        
        feed_dict = {x: imageSimple, y_true: np.zeros((1, 10)), hold_prob : 0.5 }
        classification = sess.run(tf.argmax(y_pred,1), feed_dict)
        
    return classification

def readGrayImageFromData(data):
    imageToPredict = rgba2rgb(data)
    return rgb2grey(imageToPredict)

def readGrayImageFromUrl(url):
    imageToPredict = rgba2rgb(io.imread(url))
    return rgb2grey(imageToPredict)

def simplifyImage(originalImage):
    partialResizeImage = tform.resize(originalImage, (28, 28), order=5)
    return (1 - np.reshape(partialResizeImage,newshape=(1,784)))



# print(sih.fileList[int(myPrediction)])
imageURL4 = 'https://firebasestorage.googleapis.com/v0/b/tensorweb-af554.appspot.com/o/test04.png?alt=media&token=b3a33d7e-9189-4255-9023-6ce6ba2d77df'

@app.route('/hook', methods=['POST'])
def get_image():
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)
    image_PIL = Image.open(BytesIO(base64.b64decode(image_data)))
    image_np = np.array(image_PIL)
    myPrediction = predictFromDataImage(image_np)
    print(myPrediction)
    result = { 'prediction': np.asscalar(myPrediction)}
    return Response(json.dumps(result), mimetype='application/json')

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]



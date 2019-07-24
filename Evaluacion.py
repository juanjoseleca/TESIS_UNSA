#! /usr/bin/env python
#LIBRERIAS A UTILIZAR
#==========================================================
import unittest
import re, string, unicodedata
#import contractions
#import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn
import csv
import sys
import json
import urllib
from urllib.request import urlopen
import requests 
import normalizacion
#==========================================================
#FUNCIONES DE PRUEBA
#==========================================================
def porcentaje_total(lista):
    suma=0
    for x in lista:
        suma+=x
    return suma
def prueba_clasificacion(entrada):
    print("entree")
    x_raw = [entrada]
    y_test = [0,0,0,0,0,0,0,0,0,1]
    #y_test = np.argmax(y_test, axis=1)
    print("hi")
    vocab_path = os.path.join("./runs/1544554905/checkpoints/", "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    checkpoint_file = tf.train.latest_checkpoint("./runs/1544554905/checkpoints/")
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])



    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    result = predictions_human_readable[0][1]
    resultString = ""
    if(result == '0.0'):
        resultString = "Desarrollo movil"
    elif(result == '1.0'):
        resultString = "Frontend"
    elif(result == '2.0'):
        resultString = "Backend"
    elif(result == '3.0'):
        resultString = "Servidores"
    elif(result == '4.0'):
        resultString = "Videojuegos"
    elif(result == '5.0'):
        resultString = "Realidad Virtual"
    elif(result == '6.0'):
        resultString = "Data Science"
    elif(result == '7.0'):
        resultString = "Machine Learning"
    elif(result == '8.0'):
        resultString = "Emprendimiento"
    print("String resultante:",resultString)
    return resultString



def prueba_preprocesamineto(entrada):
    salio=normalizacion.normalize([entrada])
    return salio[0]


#==========================================================
#ARGUMENTOS
#==========================================================
print(len(sys.argv))
if(len(sys.argv)==3):
    argumento_lectura=sys.argv[1]
    argumento_salida=sys.argv[2]
else:  
    argumento_lectura="https://weeblabs.com/api/references"
    argumento_salida="https://weeblabs.com/api/categories"
    
    
#==========================================================
# PARAMETROS
# ==================================================

# Parametros de Evaluacion
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

#LECTURA DE ARHIVO DE ENTRADA
#==========================================================
#LECTURA URL
#----------------------------------------------------------
if(len(sys.argv)==3):
    j = urlopen(argumento_lectura)
    jsonObj = json.load(j)
    salida_v=""
    for x in jsonObj:
        salida=x['comment'].rstrip().lstrip()
        vector=salida.split(" ")
        salida_v+=" ".join(normalizacion.normalize(vector))+"\n"
    salida_v=salida_v.rstrip()
    data=salida_v.split("\n")
    print("HERE->",data)
    x_text=data
    label_=[[1,0,0,0,0,0,0,0,0,0] for j in range(len(x_text))]
    y=np.concatenate([label_],0)
    x_raw, y_test =[x_text,y]
    y_test = np.argmax(y_test, axis=1)
#LECTURA ARHIVO
#----------------------------------------------------------
else:
    x_raw, y_test =data_helpers.cargar_test("entrada.txt")
    x_raw=normalizacion.normalize(x_raw)
    y_test = np.argmax(y_test, axis=1)
#===========================================================

# Vocabulario
#===========================================================
vocab_path = os.path.join("./runs/1544554905/checkpoints/", "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
#===========================================================

# Evaluacion
# ==========================================================
checkpoint_file = tf.train.latest_checkpoint("./runs/1544554905/checkpoints/")
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

#Comunicacion con el API
#==============================================================
API_ENDPOINT = argumento_salida
categorias=["Desarrollo movil","Frontend","Backend","Servidores","Videojuegos","Realidad Virtual","Data Science","Machine Learning","Emprendimiento","Curso Basico"]
total_=len(all_predictions)
all_predictions=all_predictions.tolist()
Porcentajes=[(all_predictions.count(0.0)*100.0)/20.0,(all_predictions.count(1.0)*100.0)/20.0,(all_predictions.count(2.0)*100.0)/20.0,(all_predictions.count(3.0)*100.0)/20.0,(all_predictions.count(4.0)*100.0)/20.0,(all_predictions.count(5.0)*100.0)/20.0,(all_predictions.count(6.0)*100.0)/20.0,(all_predictions.count(7.0)*100.0)/20.0,(all_predictions.count(8.0)*100.0)/20.0,(all_predictions.count(9.0)*100.0)/20.0]
if(len(sys.argv)==3):
    for x in range(len(categorias)):
        print(categorias[x],": ",Porcentajes[x])
        #data_api={'name':categorias[x],'weight':Porcentajes[x]}
        #r = requests.post(url = API_ENDPOINT, data = data_api)
else:
    print("Nothing")
#================================================================
salida_y=[categorias[int(i)] for i in all_predictions]
#print("what chucha pasa")
predictions_human_readable = np.column_stack((np.array(x_raw), salida_y))
out_path = os.path.join("./runs/", "..", "prediction.csv")
#print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)

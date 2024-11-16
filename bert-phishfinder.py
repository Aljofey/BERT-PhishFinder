from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input 
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding 
from keras.layers import SpatialDropout1D,RepeatVector
from tensorflow import keras
#from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Add, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D 
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling2D,Concatenate
import seaborn as sns
import requests
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
#from tcn import TCN, tcn_full_summary
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import GRU
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.layers import LSTM, Reshape, Bidirectional
from keras import initializers, regularizers, constraints, optimizers, layers, activations
from keras.layers import Activation, BatchNormalization
from keras.layers import AveragePooling1D, Conv2D, MaxPooling2D,GlobalMaxPool1D
from keras.utils import np_utils
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense, TimeDistributed)
from keras.layers import BatchNormalization, InputSpec, add
#from keras_self_attention import SeqSelfAttention
from tensorflow.keras.utils import to_categorical, custom_object_scope
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, tqdm_pandas
import math
import scipy
from keras.engine.topology import Layer
from scipy.stats import skew
from keras.initializers import *
##import librosa
##import librosa.display
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import seaborn as sns
import glob 
import os
import sys
import pickle
from tensorflow.keras.layers import Lambda
#import IPython.display as ipd  # To play sound in the notebook
import warnings
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.callbacks import EarlyStopping
#from keras_transformer import get_model, decode
from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification
import transformers
import tensorflow_hub as hub
from keras.utils.np_utils import to_categorical
#import tokenization
from bert import tokenization
from tensorflow.keras.layers import  Attention
from tensorflow.keras import backend as K
#from tensorflow_model_optimization.sparsity import keras as sparsity
import absl.flags
import csv
import itertools
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score, recall_score, roc_curve
from transformers import TFXLNetModel, XLNetTokenizer
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel
from transformers import AlbertTokenizer, TFAlbertModel
from matplotlib.ticker import ScalarFormatter
from transformers import AutoTokenizer, SwitchTransformersEncoderModel
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer 
from keras.backend import softmax
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers
import mesh_tensorflow as mtf



matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14
sns.set(font_scale=1.4)


absl.flags.FLAGS(sys.argv)

with tf.device('/GPU:0'):

  #from keras.utils import plot_model
  print(tf.test.is_gpu_available())
  print(tf.test.gpu_device_name())
  
  labels, texts = [], []
  
  #new readding
  #file_path = "/home/ali/anaconda3/phishing_site_urls.csv"  # Update with your file path
  #
  #with open(file_path, "r") as file:
  #    reader = csv.reader(file, delimiter=",")  # Set the delimiter to "\t" for tab-separated values
  #    for row in reader:
  #        texts.append(row[0])
  #        if (row[1]=='bad'):
  #               labels.append('1')
  #        else:
  #               labels.append('0')
  
  
  ##this my old reading
#  with open('/home/ali/anaconda3/dataset1.txt', 'r',encoding='latin-1') as file:
#      for line in file:
#          columns = line.strip().split('|')
#          #columns = line.strip().split('\t')
#          texts.append(columns[0])
#          if (columns[1]=='+1'):
#              labels.append('1')
#          else:
#              labels.append('0')
  
  
  # create a dataframe using texts and lables
  trainDF = pd.DataFrame()
  trainDF['text'] = texts
  trainDF['label'] =labels
  
 #this is the old spilit of dataset 80:20 befor we use 5-fold cross-validation
#  train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=0.2, random_state=0)
  
          
  #This is CNN_LSTM model that we compared it with our berft-phish finder model
  def CNN_LSTM(embedding_matrix,SEQ_LEN=200):
  
      inputs = tf.keras.layers.Input(shape = (SEQ_LEN,))
      #x1 = Embedding(vocab_size, 96)(inputs)
  
      x1 = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs)
  
      #x1 = Embedding(size_of_vocabulary, 100,trainable=True)(inputs)
      
      x1=Dropout(0.25)(x1)
      x1=Conv1D(128, 8, activation='relu')(x1)
      x1=MaxPooling1D(pool_size=2)(x1)
      x1=Conv1D(128, 10, activation='relu')(x1)
      x1=MaxPooling1D(pool_size=2)(x1)
      x1=Conv1D(256, 12, activation='relu')(x1)
      x1=MaxPooling1D(pool_size=2)(x1)
      x1=LSTM(256, return_sequences=True, recurrent_dropout=0.2)(x1)
      x1=LSTM(256, return_sequences=True, recurrent_dropout=0.2)(x1)
      output=Dense(1024,activation='relu')(x1)
      flat = Flatten()(output)    
      outputs=Dense(1,activation='sigmoid')(flat)
      model3 = Model(inputs=inputs, outputs=outputs)
      model3.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
      print(model3.summary())
      return model3
  
          
          
  def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
      """
      This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
      """
      sns.set(style="whitegrid", font_scale=1.2)
      plt.grid(False)
      plt.imshow(cm, interpolation='nearest', cmap=cmap)
      #plt.title(title)
      #plt.colorbar()
      # Change the border color and width of the outside borders
      border_color = 'black'
      border_width = 0.8
      
      # Change the border color and width of the inside borders
      inside_border_color = 'white'
      inside_border_width = 0.8
      tick_marks =np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=0,fontsize=18, fontweight='bold')
      plt.yticks(tick_marks, classes,fontsize=18, fontweight='bold')
    
      if normalize:
              cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
          #print("Normalized confusion matrix")
      else:
              1#print('Confusion matrix, without normalization')
  
              #print(cm)
  
      thresh = cm.max() / 2.
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black",fontsize=20,     fontweight='bold')
      plt.tight_layout()
      plt.ylabel('True label',fontsize=20,fontweight='bold')
      plt.xlabel('Predicted label',fontsize=20,fontweight='bold')
      
          # Set the border color and width of the outside borders
      plt.gca().spines['top'].set_color(border_color)
      plt.gca().spines['bottom'].set_color(border_color)
      plt.gca().spines['left'].set_color(border_color)
      plt.gca().spines['right'].set_color(border_color)
      
      plt.gca().spines['top'].set_linewidth(border_width)
      plt.gca().spines['bottom'].set_linewidth(border_width)
      plt.gca().spines['left'].set_linewidth(border_width)
      plt.gca().spines['right'].set_linewidth(border_width)
      
      
             
        
#this function is used to encode and tokeniz the URLs text 
  def bert_encode(texts, tokenizer, max_len):
      all_tokens = []
      all_masks = []
      all_segments = []
      
      for text in texts:
          text = tokenizer.tokenize(text)
          
          text = text[:max_len-2]
          input_sequence = ["[CLS]"] + text + ["[SEP]"]
          pad_len = max_len-len(input_sequence)
          
          tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
          pad_masks = [1] * len(input_sequence) + [0] * pad_len
          segment_ids = [0] * max_len
          
          all_tokens.append(tokens)
          all_masks.append(pad_masks)
          all_segments.append(segment_ids)
          
      return np.array(all_tokens)
      #np.array(all_masks)
      #, np.array(all_segments)
  
  
  
  # this is addtional function to encode text using the tokenizer provided
  def get_inputs(tweets, tokenizer, max_len=200):
      """ Gets tensors from text using the tokenizer provided"""
      inps = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in tweets]
      inp_tok = np.array([a['input_ids'] for a in inps])
      ids = np.array([a['attention_mask'] for a in inps])
      segments = np.array([a['token_type_ids'] for a in inps])
      return inp_tok
       #,ids, segments
       
       
  # this is also addtional dunction to encode the text using the tokenizer provided     
  def tokenize_roberta(data,tokenizer,max_len=200) :
      input_ids = []
      attention_masks = []
      for i in range(len(data)):
          encoded = tokenizer.encode_plus(
              data[i],
              add_special_tokens=True,
              max_length=max_len,
              padding='max_length',
              return_attention_mask=True
          )
          input_ids.append(encoded['input_ids'])
          attention_masks.append(encoded['attention_mask'])
      return np.array(input_ids),np.array(attention_masks)
       
       
      
  def warmup(epoch, lr):
      """Used for increasing the learning rate slowly, this tends to achieve better convergence.
      However, as we are finetuning for few epoch it's not crucial.
      """
      return max(lr +1e-6, 2e-5)      
  
  
  def plot_metrics(pred, true_labels):
      """Plots a ROC curve with the accuracy and the AUC"""
      acc = accuracy_score(true_labels, np.array(pred.flatten() >= .5, dtype='int'))
      fpr, tpr, thresholds = roc_curve(true_labels, pred)
      auc = roc_auc_score(true_labels, pred)
  
      fig, ax = plt.subplots(1, figsize=(8,8))
      ax.plot(fpr, tpr, color='red')
      ax.plot([0,1], [0,1], color='black', linestyle='--')
      ax.set_title(f"AUC: {auc}\nACC: {acc}");
      return fig
      
  #embedding_weights = []    
  
      
  #this function is used for character embeding fearures of URLs
  def charcter_embedding(texts, max_len):
      tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
      tk.fit_on_texts(texts)
      ##
      alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
      ##
      char_dict = {}
      for i, char in enumerate(alphabet):
          char_dict[char] = i + 1
      
          tk.word_index = char_dict.copy()
      # Add 'UNK' to the vocabulary
      tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
      
      sequences = tk.texts_to_sequences(texts)
      #test_texts = tk.texts_to_sequences(valid_x)
      
      ### Padding
      data = pad_sequences(sequences, maxlen=200, padding='post')
      print('Load')
      return data
      
      
  #this extended function that used for character embeding fearures of URLs used by Zhang et al. [22] method    
  def charcter_embedding2(texts, max_len):
  
      # Initialize the tokenizer for character-level embedding
      tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
      tk.fit_on_texts(texts)

      # Define character and custom token indices
      alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
      special_tokens = ["account", "admin", "administrator", "auth", "bank", "client", "confirm", "cmd", "email", 
                  "host", "login", "password", "pay", "private", "registed", "safe", "secure", "security", 
                  "sign", "service", "signin", "submit", "user", "update", "validation", "verification", "webscr"]

      char_dict = {char: i + 1 for i, char in enumerate(alphabet)}

      # Set up indices for <PAD> and <UNK>
      pad_index = len(char_dict) + len(special_tokens) + 1
      unk_index = pad_index + 1

      # Update tokenizer's word index to include custom characters and tokens
      tk.word_index = char_dict.copy()
      tk.word_index.update({token: idx + len(char_dict) for idx, token in enumerate(special_tokens)})
      tk.word_index[tk.oov_token] = unk_index
      tk.word_index['<PAD>'] = pad_index

      # Convert training and test sequences to integer sequences
      sequences = tk.texts_to_sequences(texts)
     

      #Pad the sequences to ensure consistent input size
      max_len = 200
      data = pad_sequences(sequences, maxlen=max_len, padding='post', value=pad_index)
      
      return data
      
      
  #this function is used to get weights matrix when we use character embeding fearures of URLs used by Zhang et al. [22] method  
  def get_embedding_weights2(text):
  
       # Initialize the tokenizer for character-level embedding
      tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
      tk.fit_on_texts(texts)

      # Define character and custom token indices
      alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
      special_tokens = ["account", "admin", "administrator", "auth", "bank", "client", "confirm", "cmd", "email", 
                  "host", "login", "password", "pay", "private", "registed", "safe", "secure", "security", 
                  "sign", "service", "signin", "submit", "user", "update", "validation", "verification", "webscr"]

      char_dict = {char: i + 1 for i, char in enumerate(alphabet)}

      # Set up indices for <PAD> and <UNK>
      pad_index = len(char_dict) + len(special_tokens) + 1
      unk_index = pad_index + 1

      # Update tokenizer's word index to include custom characters and tokens
      tk.word_index = char_dict.copy()
      tk.word_index.update({token: idx + len(char_dict) for idx, token in enumerate(special_tokens)})
      tk.word_index[tk.oov_token] = unk_index
      tk.word_index['<PAD>'] = pad_index
      vocab_size = len(tk.word_index) + 1  # +1 for 0 index padding
      embedding_weights2 = []
      embedding_weights2 = np.zeros((vocab_size, vocab_size))

      for char, idx in tk.word_index.items():
          if idx < vocab_size:
              embedding_weights2[idx, idx] = 1.0

      embedding_weights2 = np.array(embedding_weights2)
      print("Embedding weights2 shape:", embedding_weights2.shape)
      print("Vocab size2:", vocab_size)
      return embedding_weights2
      
        
  
  #this function is used to get weights matrix when we use character embeding fearures of URLs used by Aljofey et al. [8] method  
  def get_embedding_weights(text):
            #character embeding fearures
       tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
       tk.fit_on_texts(text)
       ##
       alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
       ##
       char_dict = {}
       for i, char in enumerate(alphabet):
           char_dict[char] = i + 1
       
           tk.word_index = char_dict.copy()
       # Add 'UNK' to the vocabulary
       tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
       vocab_size = len(tk.word_index)+1
                     # Embedding weights
       embedding_weights = []  # (70, 69)
       embedding_weights.append(np.zeros(vocab_size))  # (0, 69)
     
       for char, i in tk.word_index.items():  # from index 1 to 69
         onehot = np.zeros(vocab_size)
         onehot[i - 1] = 1
         embedding_weights.append(onehot)
     
       embedding_weights = np.array(embedding_weights)
       return embedding_weights
  
  
  
  
  # label encode the target variable 
#  encoder = preprocessing.LabelEncoder()
#  labels_data=encoder.fit_transform(labels)
#  train_y = encoder.fit_transform(train_y)
#  valid_y = encoder.fit_transform(valid_y)
#  train_y1 = to_categorical(train_y)
#  valid_y1 = to_categorical(valid_y)
 
  
  max_len = 200
  
  from transformers import TFDistilBertModel
  from transformers import DistilBertTokenizer
  from transformers import DistilBertConfig
  


  num_layers = 6
  num_attention_heads = 12
  #dropout_rate = 0.1
  
  #we use this when we try to change the layers and attention_heads of distilbert to make new config  
#  config = DistilBertConfig(
#      num_hidden_layers=num_layers,
#      #num_attention_heads=num_attention_heads
#  )

    # this  if we need to make new config
  #transformer_layer=TFDistilBertModel(config)
  #  dummy_input = tf.constant([[0, 1, 2]])  # Dummy input to initialize model weights
#  _ = transformer_layer(dummy_input)
#
#   #Access the embedding matrix of the new in case of new configuration of distilbert... 
#  embedding_matrix = transformer_layer.get_input_embeddings().weight.numpy()
  
  from transformers import TFAutoModel, AutoTokenizer

  
  tokenizer_DistilBERT = transformers.DistilBertTokenizer.from_pretrained('bert-base-uncased')
  transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
#  train_input = bert_encode(train_x, tokenizer_DistilBERT, max_len=max_len)
#  test_input = bert_encode(valid_x, tokenizer_DistilBERT, max_len=max_len
  
  #this for xlnet model
#  xlnet_model = 'xlnet-base-cased'
#  xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
  #inp_train = get_inputs(train_x, xlnet_tokenizer)
  #inp_test = get_inputs(valid_x, xlnet_tokenizer)
  
  #this for roberta model
  #tokenizer_roberta = RobertaTokenizerFast.from_pretrained("distilroberta-base")
  #train_input_ids = bert_encode(train_x,tokenizer_roberta)
  #val_input_ids = bert_encode(valid_x,tokenizer_roberta)
  #roberta_model = TFRobertaModel.from_pretrained('distilroberta-base')
  
  #this for Albert model
  #albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
  #albert_model = TFAlbertModel.from_pretrained('albert-base-v2')
  #train_input_ids = bert_encode(train_x,albert_tokenizer)
  #val_input_ids = bert_encode(valid_x,albert_tokenizer)
  
  
  
  #this to get the embedding_matrix weights matricx from distilbert that we need to embedding it for other models like CNN and etc
  embedding_matrix = transformer_layer.weights[0].numpy()
  
  

  
  #this function is to create both of ropberta and albert models to compar..., we just need to change the parametrs...
  def create_roberta_model(bert_model, max_len=200):
      
      input_ids = Input(shape=(max_len,),dtype='int32')
      #attention_masks = Input(shape=(max_len,),dtype='int32')
      output = bert_model(input_ids)
      output = output[1]
      output = Dense(1, activation='sigmoid', name='outputs')(output)
      model = Model(inputs = [input_ids],outputs = output)
      opt = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-7)
      model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])
      print(model.summary()) 
      return model
  
  
  #this function is to create XLnet model that we compared with our model...
  def create_xlnet(mname):
      """ Creates the model. It is composed of the XLNet main block and then
      a classification head its added
      """
      # Define token ids as inputs
      word_inputs = tf.keras.Input(shape=(200,), name='word_inputs', dtype='int32')
  
      # Call XLNet model
      xlnet = TFXLNetModel.from_pretrained(mname)
      xlnet_encodings = xlnet(word_inputs)[0]
  
      # CLASSIFICATION HEAD 
      # Collect last step from last hidden state (CLS)
      doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
      # Apply dropout for regularization
      doc_encoding = tf.keras.layers.Dropout(.1)(doc_encoding)
      # Final output 
      outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(doc_encoding)
  
      # Compile model
      model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
      model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
      xlnet.summary()
  
      return model
      
  
  
  # this function is to build the Bert_PhishFinder_model.... 
  def build_Bert_PhishFinder_model(transformer,embedding_matrix,max_len):
      N_SAMPLES = 8 
      input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
      print("input_ids",input_ids.shape)
      #input_ids2 = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids2")
      #input_ids2 = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids2")
      attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
      #segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
      
      last_hidden_state = transformer(input_ids)[0]
#      output = tf.keras.layers.Dense(1, activation="sigmoid")(last_hidden_state)  # Binary classification (phishing/benign)
      
      
      #clf_output = last_hidden_state[:, 0, :]
  
      #apply SpatialDropout1D along with GlobalMaxPooling1D and GlobalAveragePooling1D 
      clf_output = SpatialDropout1D(0.1)(last_hidden_state)
      x_avg = layers.GlobalAveragePooling1D()(clf_output)
      x_max = layers.GlobalMaxPooling1D()(clf_output)
      x = layers.Concatenate()([x_avg, x_max])  
      samples = []
      for n in range(N_SAMPLES):  # Ensure the 'for' loop is correctly aligned
          sample_mask = layers.Dense(64, activation='relu', name=f'dense_{n}')
          sample = layers.Dropout(0.5)(x)
          sample = sample_mask(sample)
          sample = layers.Dense(1, activation='sigmoid', name=f'sample_{n}')(sample)
          samples.append(sample)

#      # Average the outputs of all samples
      output = layers.Average(name='output')(samples)

      #dense = tf.keras.layers.Dense(256, activation='relu')(merged)
      #drop=Dropout(0.5)(dense)
      #out = tf.keras.layers.Dense(1, activation='sigmoid')(clf_output)
      
      model = Model(inputs=[input_ids], outputs=output)
      model.compile(Adam(lr=1e-5),loss='binary_crossentropy',metrics=['accuracy'])
      print(model.summary()) 
      return model
      

      
  #here is the TCN model that we used for comparsions with our bert-phish finder....
  def TCN_1(embedding_matrix,length=200,kernel_size = 3, activation='relu'):
      inp = Input( shape=(length,))
      x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=True)(inp)
      x = SpatialDropout1D(0.1)(x)
      dilations = [1, 2, 4, 8, 16]
  
  # Define the TCN model using Conv1D layers
      for dilation_rate in dilations:
          x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=dilation_rate, activation='relu', padding='causal')(x)
  ##
  
  ##    x = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = activation, name = 'tcn1')(x)
  ##    x = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = activation, name = 'tcn2')(x)
      avg_pool = GlobalAveragePooling1D()(x)
      max_pool = GlobalMaxPooling1D()(x)
      conc = concatenate([avg_pool, max_pool])
      conc = Dense(16, activation="relu")(conc)
      conc = Dropout(0.1)(conc)
      outp = Dense(1, activation="sigmoid")(conc)
      model = Model(inputs=inp, outputs=outp)
      model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
      print(model.summary()) 
      return model
  
   
  # aljofey_2020 et al. [8] method
  def aljofey_2020 (embedding_weights1,vocab_size=95):
                inputs = tf.keras.layers.Input(shape = (200,))
                x = Embedding(vocab_size + 1, 96, weights=[embedding_weights1])(inputs)              
                conv_layers = [[256, 7, 3],
                     [256, 7, 3],
                     [256, 3, -1],
                     [256, 3, -1],
                     [256, 3, -1],
                     [256, 3, -1],
                     [256, 3, 3]]

                fully_connected_layers = [2028, 2048]
                dropout_p = 0.5
                #optimizer = 'adam'
                for filter_num, filter_size, pooling_size in conv_layers:
                    x = Conv1D(filter_num, filter_size)(x)
                    x = Activation('relu')(x)
                    if pooling_size != -1:
                        x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
                x = Flatten()(x)  # (None, 8704)

                for dense_size in fully_connected_layers:
                  x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
                  x = Dropout(dropout_p)(x)
# Output Layer
                out = Dense(1,activation='sigmoid')(x)
# Build model
                model = Model(inputs=inputs, outputs=out)
                model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
                print(model.summary())
                return model
                

  # Wei_2020 et al. [24] method
  def wei_wei (embedding_weights1,vocab_size=95):
              inputs2 = tf.keras.layers.Input(shape=(200,))
              x4 = Embedding(vocab_size + 1, 96, weights=[embedding_weights1])(inputs2)
              x4 = Conv1D(64, kernel_size=8, activation='relu')(x4)
              x4 = MaxPooling1D(pool_size=2)(x4)

              x4 = Conv1D(16, kernel_size=16, activation='relu')(x4)
              x4 = MaxPooling1D(pool_size=2)(x4)

              x4 = Conv1D(8, kernel_size=32, activation='relu')(x4)
              x4 = MaxPooling1D(pool_size=2)(x4)

              x4 = Flatten()(x4)  # Add this line to flatten the output

              out = Dense(32, activation='relu')(x4)
              out = Dense(1, activation='sigmoid')(out)

              model = Model(inputs=inputs2, outputs=out)
              model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
              print(model.summary())
              return model
              
              
  # Alshehri et al. [23] method
  def Mohammed_2022 (embedding_weights1,vocab_size=95):
              inputs2 = tf.keras.layers.Input(shape = (200,))
              x4 = Embedding(vocab_size + 1, 96, weights=[embedding_weights1])(inputs2)
              c1=Conv1D (128, kernel_size=4, activation='relu')(x4)
              c2=Conv1D (128, kernel_size=6, activation='relu')(x4)
              c3=Conv1D (128, kernel_size=10, activation='relu')(x4)
              c4=Conv1D (128, kernel_size=20, activation='relu')(x4)

              flat1 = Flatten()(c1)
              flat2 = Flatten()(c2)
              flat3 = Flatten()(c3)
              flat4 = Flatten()(c4)
              flat5 = Flatten()(x4)

              concatenated1 = Concatenate()([flat1,flat2,flat3,flat4,flat5])

              concatenated1=Dropout(0.5)(concatenated1)

              concatenated1 = Dense(64, activation='relu')(concatenated1)
              concatenated1 = Dense(64, activation='relu')(concatenated1)
              concatenated1 = Dense(64, activation='relu')(concatenated1)

              logits = layers.Dense(1, activation='sigmoid')(concatenated1)
              model=Model(inputs=inputs2, outputs=logits)
                        #model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
              model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
              print(model.summary())
              return model
              
              
   # Zhang_2021 et al. [23] method     
  def CNN_biLSTM(embedding_weights1,vocab_size=123):  
            inputs = tf.keras.layers.Input(shape = (200,))
            x1 = Embedding(vocab_size + 1, 124, weights=[embedding_weights1])(inputs)
            
            x1=Dropout(0.25)(x1)
            x1=Conv1D(128, 8, activation='relu')(x1)  
            x1=Conv1D(128, 10, activation='relu')(x1)
            x1=MaxPooling1D(pool_size=2)(x1)
            x1=Conv1D(256, 12, activation='relu')(x1)
            x1=MaxPooling1D(pool_size=2)(x1)
            x1=LSTM(64, return_sequences=True, recurrent_dropout=0.2)(x1)
            x1=MaxPooling1D(pool_size=2)(x1)
            x1=LSTM(128, return_sequences=True, recurrent_dropout=0.2)(x1)
            output=Dense(1024,activation='relu')(x1)
            flat = Flatten()(output)    
            outputs=Dense(1,activation='sigmoid')(flat)
            model3 = Model(inputs=inputs, outputs=outputs)
            #model3.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
            model3.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
            print(model3.summary())
            return model3
            
       
   # Hussain_2023 et al. [25] method               
  def CNN_Fusion(embedding_weights1,vocab_size=95):
  
      inputs = tf.keras.layers.Input(shape = (200,))
      x1 = Embedding(*embedding_matrix1.shape, weights=[embedding_weights1],trainable=False)(inputs)
      #x1 = Embedding(vocab_size + 1, 96, weights=[embedding_weights1])(inputs)
      
      cnn = Conv1D(filters=128, kernel_size=8, padding='same',activation='relu')(x1)
      cnn = SpatialDropout1D(0.4)(cnn)
      cnn = GlobalMaxPooling1D()(cnn) 
      
      cnn1 = Conv1D(filters=128, kernel_size=10, padding='same',activation='relu')(x1)
      cnn1 = SpatialDropout1D(0.4)(cnn1)
      cnn1 = GlobalMaxPooling1D()(cnn1) 
      
      cnn3 = Conv1D(filters=256, kernel_size=12, padding='same',activation='relu')(x1)
      cnn3 = SpatialDropout1D(0.4)(cnn3)
      cnn3 = GlobalMaxPooling1D()(cnn3)
      
      concatenated = Concatenate()([cnn, cnn1, cnn3])
      output = Dense(128, activation='relu')(concatenated)
      #flat1 = Flatten()(output)
      #concatenated = Concatenate()([flat1, flat2])
      output=Dropout(0.4)(output)
      output = Dense(1, activation='sigmoid')(output)
  
      model3 = Model(inputs=inputs, outputs=output)
      model3.compile(loss='binary_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
      print(model3.summary())
      return model3
  
  

  
 #this for using 5-fold cross validation and spilit the datasets and save them to folders... 
  from sklearn.model_selection import StratifiedKFold
 
#   #Ensure the directory for saving splits exists
#  output_dir = 'Dataset1_kfold_splits/'
#  if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#        
#   #Cross-validation loop
#  #kf = KFold(n_splits=5, shuffle=True, random_state=42)
#  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#   
#  #trainDF['text'] 
#  
#  texts = np.array(texts)
#  #labels_data = np.array(labels_data)
#
#  from tensorflow.keras.callbacks import EarlyStopping
#  early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
#
#  fold_no = 1
#  for train_index, val_index in skf.split( texts , labels_data):
#      print(f'Fold {fold_no}')
#      # Split data into training and validation sets
#      
#      X_train_input_ids, X_val_input_ids = texts[train_index], texts[val_index]
#      y_train, y_val = labels_data[train_index], labels_data[val_index]
      
#      # Save training data to CSV
#      train_df = pd.DataFrame({'X_train_input_ids': X_train_input_ids, 'y_train': y_train})
#      train_file = os.path.join(output_dir, f'train_fold_{fold_no}.csv')
#      train_df.to_csv(train_file, index=False)
#
#      # Save validation data to CSV
#      val_df = pd.DataFrame({'X_val_input_ids': X_val_input_ids, 'y_val': y_val})
#      val_file = os.path.join(output_dir, f'val_fold_{fold_no}.csv')
#      val_df.to_csv(val_file, index=False)
#      
#      fold_no += 1
      
  def training():
      fold_no = 1
      print(f'Running ds1 method... Fold {fold_no}')
    
      # Load training data , we can change only the name of dataset path and fold_no to load it....
      train_file = f'Dataset1_kfold_splits/train_fold_{fold_no}.csv'
      train_df = pd.read_csv(train_file)
      X_train_input_ids = train_df['X_train_input_ids'].values
      y_train = train_df['y_train'].values

      # Load validation data, we can change only the name of dataset path adn fold_no to load it....
      val_file = f'Dataset1_kfold_splits/val_fold_{fold_no}.csv'
      val_df = pd.read_csv(val_file)
      X_val_input_ids = val_df['X_val_input_ids'].values
      y_val = val_df['y_val'].values
      
      
      
#      #here were extract charcter features of URL within each fold of the dataset to avoid the leak of features...
#      train_data=charcter_embedding(X_train_input_ids,max_len)
#      test_data=charcter_embedding(X_val_input_ids,max_len)
#      embedding_weights1=get_embedding_weights(X_train_input_ids)
#
   #here were extract charcter features2 of URL of zhange et al. [22] method within each fold of the dataset to avoid the leak of features...
   
#      train_data=charcter_embedding2(X_train_input_ids,max_len)
#      test_data=charcter_embedding2(X_val_input_ids,max_len)
#      embedding_weights2=get_embedding_weights2(X_train_input_ids)
      
      
      #here were extract token features of URL using xlnet_tokenizer...within each fold  
#      train_input=bert_encode(X_train_input_ids, xlnet_tokenizer, max_len)
#      test_input=bert_encode(X_val_input_ids, xlnet_tokenizer, max_len) 


      #here were extract token features of URL using albert_tokenizer...within each fold  
#      train_input=bert_encode(X_train_input_ids, albert_tokenizer, max_len)
#      test_input=bert_encode(X_val_input_ids, albert_tokenizer, max_len) 


      #here were extract token features of URL using tokenizer_roberta...within each fold  
#      train_input=bert_encode(X_train_input_ids, tokenizer_roberta, max_len)
#      test_input=bert_encode(X_val_input_ids, tokenizer_roberta, max_len) 

      #here were extract token features of URL using tokenizer_DistilBERT...within each fold  
      train_input=bert_encode(X_train_input_ids, tokenizer_DistilBERT, max_len)
      test_input=bert_encode(X_val_input_ids, tokenizer_DistilBERT, max_len)
      
#      
#       # Build and train the models
        #Function to build the TCN model
      #model=TCN_1(embedding_matrix)
      
        #Function to build the hybrid CNN-LSTM model
      #model=CNN_LSTM(embedding_matrix)
      
        #Function to build both of  Hussain_2023 et al. [25] and Multi-Scale CNN methods the same function... 
      #model=CNN_Fusion(embedding_matrix)
      
      
       #build Zhang_2021 et al. [23] method  
      #model=CNN_biLSTM(embedding_weights2)
      
      
         #build Alshehri et al. [23] method 
      #model=Mohammed_2022(embedding_weights1)
      
      
       #Build Wei_2020 et al. [24] method
      #model=wei_wei(embedding_weights1)
      
        #Build aljofey_2020 et al. [8] method 
      #model=aljofey_2020(embedding_weights1)
      
        #Build XLNet [38] model
      #model = create_xlnet(xlnet_model)
      
       #this function to build both RoBERTa [29] and ALBERT [39] models but we have to change the parametrs etc...
      #model=create_roberta_model(albert_model)
      #model = create_roberta_model(roberta_model) 
      
      #Build  Bert_PhishFinder_model
      model = build_Bert_PhishFinder_model(transformer_layer,embedding_matrix,max_len)
       # Set up EarlyStopping callback
      early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
      checkpoint = ModelCheckpoint('ds1fold1.h5', monitor='val_accuracy', save_best_only=True)

      history=model.fit([train_input], y_train, epochs=50, batch_size=64,
             callbacks=[checkpoint,early_stopping], validation_data=([test_input], y_val), verbose=2)
      model.load_weights('ds1fold1.h5')
      # evaluate model on training dataset
      loss, acc = model.evaluate([train_input], array(y_train), verbose=0)
      print('Train Accuracy: %f' % (acc*100))
      loss, acc = model.evaluate([test_input], array(y_val), verbose=0)
      print('our model: Test Accuracy: %f' % (acc*100))
      
      predicted = model.predict([test_input])
      t = [1 if np.any(prob > 0.5) else 0 for prob in predicted]
      print(metrics.classification_report(y_val, t))
      print("\n f1_score(in %):", metrics.f1_score(y_val, t)*100)
      print("model accuracy(in %):", metrics.accuracy_score(y_val, t)*100)
      print("precision_score(in %):", metrics.precision_score(y_val,t)*100)
      print("roc_auc_score(in %):", metrics.roc_auc_score(y_val,t)*100)
      print("recall_score(in %):", metrics.recall_score(y_val,t)*100)
      
#      auc = roc_auc_score(y_val, t)
#      print("auc:\n",auc)
  
      acc = history.history['accuracy']
      val_acc = history.history['val_accuracy']
      loss = history.history['loss']
      val_loss = history.history['val_loss']
      x = range(1, len(acc) + 1)
      print("acc list:\n")
      for i in acc:
           print(",",i)
           
      print("val_acc list:\n")
      for i in val_acc:
           print(",",i)
           
      print("loss list:\n")
           
      for i in loss:
           print(",",i)
           
      print("val loss:\n")     
      
      for i in val_loss:
           print(",",i)
                
      print("x range list:\n")
      
      for i in x:
           print(",",i) 
           
      from sklearn.metrics import confusion_matrix
      cnf_matrix_tra = confusion_matrix(y_val, t)
      print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
      class_names = [0,1]
  
      print("cnf_matrix_tra:\n")
  
      for i in cnf_matrix_tra:
          print(",",i) 
      print("cnf_matrix_tra 2:\n") 
  
      print(cnf_matrix_tra)  
       
          
  
  
 
      
  #this function that we use to extract the features, build and train the models....    
  training()
 
 #we use this to select the optimal leaning rate...
 # Define the learning rate scheduler
  #start_lr = 1e-5
  #end_lr = 6e-5
  #num_epochs = 50
  ### Define the learning rate scheduler
  #def lr_scheduler(epoch, lr):
  #    return lr + (end_lr - start_lr) / num_epochs
  ##
  #lr_finder = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=2)
  #lr_values = []
  #accuracy_values = []
  ### Get the LR and loss values at each epoch
  #for epoch, lr in enumerate(history.history['lr']):
  #    lr_values.append(lr)
  #    accuracy_values.append(history.history['accuracy'][epoch]) 
  ##    
  #best_lr_index = accuracy_values.index(min(accuracy_values))
  #best_lr = lr_values[best_lr_index]
  ##
  #print("Best Learning Rate:", best_lr) 
  #print("lr_values list:\n")
  #for i in lr_values:
  #     print(",",i)
  #     
  #print("accuracy_values:\n")
  #for i in accuracy_values:
  #     print(",",i)   
               
  #sns.set(style="whitegrid", font_scale=1.2)          
  #fig, ax = pyplot.subplots(figsize=(6,6))
  ##Plot the LR vs. Loss curve with a smooth line
  #ax.plot(lr_values, accuracy_values,color='red')
  # #Set labels and title
  #ax.set_xlabel('Learning Rate',fontsize=18,fontweight='bold')
  #ax.set_ylabel('Accuracy',fontsize=18,fontweight='bold')
  #ax.set_title('Learning Rate Finder',fontsize=18,fontweight='bold')
  ##pyplot.xscale('log')
  # #Format the x-axis to display values in scientific notation
  #ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
  #ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  # #Add gridlines
  #ax.grid(True, linestyle='--', alpha=0.5)
  # #Add a legend
  #legend=ax.legend(['Accuracy'], loc='lower right',fontsize=18)
  #for text in legend.get_texts():
  #    text.set_fontweight('bold')
  ## Set the border color and width
  #border_color = 'black'
  #border_width = 0.8
  ## Set the spines (borders) to be visible
  #ax.spines['top'].set_visible(True)
  #ax.spines['bottom'].set_visible(True)
  #ax.spines['left'].set_visible(True)
  #ax.spines['right'].set_visible(True)
  ## Set the color and width of each spine
  #ax.spines['top'].set_color(border_color)
  #ax.spines['bottom'].set_color(border_color)
  #ax.spines['left'].set_color(border_color)
  #ax.spines['right'].set_color(border_color)
  #ax.spines['top'].set_linewidth(border_width)
  #ax.spines['bottom'].set_linewidth(border_width)
  #ax.spines['left'].set_linewidth(border_width)
  #ax.spines['right'].set_linewidth(border_width)
  # #Adjust the spacing
  #fig.tight_layout()
  # #Save the figure as a high-resolution image (optional)
  ##fig.savefig('learning_rate_finder2.png', dpi=300)
  # #Show the figure
  #pyplot.grid(True)
  #pyplot.yticks(fontsize=18, fontweight='bold')
  #pyplot.xticks(fontsize=18,fontweight='bold') 
  #pyplot.show()
  
  
  
  sns.set(style="whitegrid", font_scale=1.2)
  
  def plot_history(history):
      acc = history.history['accuracy']
      val_acc = history.history['val_accuracy']
      loss = history.history['loss']
      val_loss = history.history['val_loss']
      x = range(1, len(acc) + 1)
      
      
      import matplotlib.pyplot as plt
      from matplotlib.font_manager import FontProperties
      
      fig, ax = plt.subplots(figsize=(8, 8))
      ax.plot(x, acc, 'b', label='Training acc',linewidth=4)
      ax.plot(x, val_acc, 'r', label='Validation acc',linewidth=4)
      
      
      # Customize labels and ticks
      ax.set_ylabel('Accuracy', fontsize=21, fontweight='bold')
      ax.set_xlabel('Epochs', fontsize=21, fontweight='bold')
      #ax.tick_params(axis='both', which='major', labelsize=18,weight='bold')
      
       # Customize legend
      legend=plt.legend(loc='lower right',fontsize=18)
      for text in legend.get_texts():
          text.set_fontweight('bold')
      
      # Customize spines (borders)
      border_color = 'black'
      border_width = 0.8
      for spine in ax.spines.values():
          spine.set_visible(True)
          spine.set_color(border_color)
          spine.set_linewidth(border_width)
          
          
      # Hide top and right spines (borders)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
          
      plt.yticks(fontsize=18, fontweight='bold')
      plt.xticks(fontsize=18,fontweight='bold')
      
      plt.grid(False)
      
      plt.show()
  
  
  
  
  plot_history(history)
  
  
 

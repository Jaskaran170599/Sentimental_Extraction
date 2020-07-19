import tensorflow as tf
from transformers import *
import tokenizers

import numpy as np
import sentencepiece as spm
import sentencepiece_pb2 as spt
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

config={
"MAX_LEN" : 96,
"TOKENIZER" : tokenizers.ByteLevelBPETokenizer("./model/vocab.json",
                                               "./model/merges.txt",
                                               lowercase=True)}

def get_model(length=128):
    ids=tf.keras.layers.Input(shape=(length,),name="input_ids",dtype=tf.int32)
    token_ids=tf.keras.layers.Input(shape=(length,),name="token_type_ids",dtype=tf.int32)    
    att_mask=tf.keras.layers.Input(shape=(length,),name="attention_mask",dtype=tf.int32)
    
    config = RobertaConfig.from_pretrained("./model/", output_hidden_states=True)
    BL=TFRobertaModel(config=config)
    
    _,_,layers=BL(ids,attention_mask=att_mask,token_type_ids=token_ids,training=False)
    drop=tf.keras.layers.Dropout(0.1)(layers[-1])
    start=tf.keras.layers.Dense(2,activation="sigmoid")(drop)
    model=tf.keras.models.Model(inputs=[ids,token_ids,att_mask],outputs=[start])
    
    
    model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5))
    
    return model

model=get_model(config["MAX_LEN"])
model.load_weights("./model/model")

def get_target(text,sentiment):
    text = " "+" ".join(str(text).split())
    
    encoded_text=config["TOKENIZER"].encode(text)
    ids=encoded_text.ids

    sentiment_id = {
    'positive': 1313,
    'negative': 2430,
    'neutral': 7974
    }

    ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + ids + [2]
    type_ids = [0, 0, 0, 0] + [0] * (len(ids)-4)
    attention = [1] * len(type_ids)
        
    if(len(ids)>config["MAX_LEN"]):
        ids=ids[:config["MAX_LEN"]-5]
        type_ids=type_ids[:config["MAX_LEN"]-5]
        attention=attention[:config["MAX_LEN"]-5]
        
    pad=config["MAX_LEN"]-len(ids)

    ids=np.array(ids+[0]*pad).reshape((1,-1))
    type_ids=np.array(type_ids+[0]*pad).reshape((1,-1))
    attention=np.array(attention+[0]*pad).reshape((1,-1))

    return {"orig":text,"input_ids":ids,"token_type_ids":type_ids,"attention_mask":attention,"sentiments":sentiment}
    
def get_text(text,pred,sentiments):
        pred=np.argmax(pred,axis=1)[0]
        
        if(len(text.split())<2):
            return text
            
        t=config["TOKENIZER"].encode(text).offsets
        t=[(0, 0)] * 4 + t + [(0, 0)]
        i,j=pred[0],pred[1]
        return text[t[i][0]:t[j][1]+1]
        
def get_result(text,sentiment):
    preprocess=get_target(text,sentiment)
    pred=model.predict(preprocess)
    return get_text(text,pred,sentiment).strip()


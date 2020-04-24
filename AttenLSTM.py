import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
# from keras.layers import RepeatVector, Dense, Activation, Lambda
# from keras.optimizers import Adam
# from keras.utils import to_categorical
# from keras.models import load_model, Model
# import keras.backend as K
import numpy as np
# from sklearn.model_selection import train_test_split

import random
from tqdm import tqdm

import unicodedata
import re
import numpy as np
import os
import io
import time

from matplotlib.pyplot import figure, cm
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

from SemanticModel import SemanticModel
from stimulus_utils import load_grids_for_stories
from stimulus_utils import load_generic_trfiles
from dsutils import make_word_ds, make_phoneme_ds


def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words= vocab, filters='')
  lang_tokenizer.fit_on_texts(lang) #come up with the dict for word to tokens and the number of tokens need for the entire dataset
  tensor = lang_tokenizer.texts_to_sequences(lang) #coverts the words to token numbers while preserving the previous structure of the dataset
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post') #pads the uneven sequences. 

  return tensor, lang_tokenizer

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, inp_embedding_matrix):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights = [inp_embedding_matrix])
    self.LSTM = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state,_ = self.LSTM(x)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz,tar_embedding_matrix ):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[tar_embedding_matrix])
    self.LSTM = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state,_ = self.LSTM(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([tar_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

if __name__ == "__main__":
    '''Load semantic model'''
    eng1000 = SemanticModel.load("english1000sm.hf5")

    '''load Stimulus Data '''

    # These are lists of the stories
    # Rstories are the names of the training (or Regression) stories, which we will use to fit our models
    Rstories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy', 
                'life', 'myfirstdaywiththeyankees', 'naked', 
                'odetostepfather', 'souls', 'undertheinfluence']

    # Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
    Pstories = ['wheretheressmoke']

    allstories = Rstories + Pstories
    grids = load_grids_for_stories(allstories)
    trfiles = load_generic_trfiles(allstories)
    wordseqs = make_word_ds(grids, trfiles) # dictionary of {storyname : word DataSequence}
    phonseqs = make_phoneme_ds(grids, trfiles) # dictionary of {storyname : phoneme DataSequence}

    '''LSTM MODEL '''
    vocab = (len(eng1000.vocab))
    window = 20
    embedding_dim = 985
    units = 985
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000 # cause tensorflow can shuffle infintie dataset it doesnt load everything into memory. Hence tell it to shuffle between 10000 datapoints at a time. 

    '''TrnDataset'''
    flatten = lambda l: [item for sublist in l for item in sublist]
    make_inputs = lambda l,window:[['<start>']+l[i-window:i]+['<end>'] for i in range(0+window,len(l),window)]
    make_outputs = lambda l,window:[['<start>']+l[i-window:i-1]+['<end>']  for i in range(1+window,len(l),window)]

    trnslices = [] # list of list 
    for i in allstories:
        trnslices.append(wordseqs[i].data)
        
    inputs = make_inputs(flatten(trnslices), window)
    targets = make_outputs(flatten(trnslices), window)

    input_tensor, inp_lang_tokenizer = tokenize(inputs)
    target_tensor, tar_lang_tokenizer = tokenize(targets)

    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1] # Calculate max_length of the target tensors
    # Creating training and validation sets using an 80-20 split
    split = 0.8
    input_tensor_train = input_tensor[:int(len(input_tensor)*split)]
    input_tensor_val = input_tensor[int(len(input_tensor)*split):]
    target_tensor_train = target_tensor[:int(len(target_tensor)*split)]
    target_tensor_val = target_tensor[int(len(target_tensor)*split):]

    # Show length
    print('len input_tensor_train :', len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    # BUFFER_SIZE = len(input_tensor_train)
    # BATCH_SIZE = 64

    # embedding_dim = 256
    # units = 1024
    vocab_inp_size = len(inp_lang_tokenizer.word_index)+1
    vocab_tar_size = len(tar_lang_tokenizer.word_index)+1

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset))

    '''Enocder and Decoder'''
    inp_embedding_matrix = np.zeros((vocab_inp_size, eng1000.data.shape[0]))
    error = []
    for word, i in inp_lang_tokenizer.word_index.items():
        try:
            embedding_vector = eng1000[word]
            if embedding_vector is not None:
                inp_embedding_matrix[i] = embedding_vector
        except KeyError as e:
                inp_embedding_matrix[i] = np.zeros((eng1000.data.shape[0]))

    tar_embedding_matrix = np.zeros((vocab_tar_size, eng1000.data.shape[0]))
    error = []
    for word, i in tar_lang_tokenizer.word_index.items():
        try:
            embedding_vector = eng1000[word]
            if embedding_vector is not None:
                tar_embedding_matrix[i] = embedding_vector
        except KeyError as e:
                tar_embedding_matrix[i] = np.zeros((eng1000.data.shape[0]))

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, inp_embedding_matrix)

    # sample input
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE,tar_embedding_matrix)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                        sample_hidden, sample_output)

    print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


    '''Loss Function and Optimiser '''
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)

    '''Training Loop'''
    EPOCHS = 10
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    












    

    












   

import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2

        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        # tf.keras.layers.GRU自动匹配cpu、gpu
        #将词转换成embedding的形式
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim=embedding_dim,
                                                    weights = [embedding_matrix], trainable = False) 

        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        self.rnn = tf.keras.layers.SimpleRNN(units = self.enc_units, return_sequences=True, return_state=True)
        self.gru = tf.keras.layers.GRU(units = self.enc_units, return_sequences=True, return_state=True)
        self.lstm = tf.keras.layers.LSTM(units = self.enc_units, return_sequences=True, return_state=True)

        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        print("x shape is ", x.shape)
        x = self.embedding(x)
        print("x after shape is ", x.shape)
        # hidden shape=[batch_sz, 2*self.enc_units] = [256, 512] 
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))


# vocab_inp_size, embedding_dim, units, BATCH_SIZE = 30000, 256, 512, 256
# encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
# sample_hidden = encoder.initialize_hidden_state()
# example_input_batch = tf.zeros(shape = (256,200))
# sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
# print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
# print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
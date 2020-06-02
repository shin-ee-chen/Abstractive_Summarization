import tensorflow as tf

class Test:
    def __init__(self, vocab_size, embedding_dim):
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim) 

    def call(self, x):
         x = self.embedding(x)
         return x

enc = Test(20000, 256)
x = tf.zeros([256, 200])
print(x.shape)
t = enc.call(x)
print(t.shape)
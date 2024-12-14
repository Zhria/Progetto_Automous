#File per definire una rete neurale dati i numero di input e di output.
import tensorflow as tf
from keras import layers

#Creiamo una classe che estende tf.keras.Model per definire la nostra rete neurale.
class ReteNeurale(tf.keras.Model):
    def __init__(self, dim_in,dim_out,softmax=False):
        super(ReteNeurale, self).__init__()
        self.softmax=softmax
        # Layer convoluzionali per estrarre caratteristiche
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', data_format="channels_last", input_shape=dim_in)
        self.pool1 = layers.MaxPooling2D((2, 2))  # Riduce la dimensione dell'immagine
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()  # Appiattisce (64, 64, 3) in (64*64*3)
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(dim_out)
    def call(self, inputs):
        #Controllo che l'input sia un tensore.
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)        
        x = self.dense2(x)
        x=self.dense3(x)
        if(self.softmax):
            #converto in float64 per evitare problemi di precisione
            x=tf.dtypes.cast(x,tf.float64)
            x=tf.nn.softmax(x,axis=1)
            #Rconverto in float32 per evitare problemi di precisione
            x=tf.dtypes.cast(x,tf.float32)
        return x

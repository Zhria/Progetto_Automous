#File per definire una rete neurale dati i numero di input e di output.
import tensorflow as tf
from keras import layers

#Creiamo una classe che estende tf.keras.Model per definire la nostra rete neurale.
class ReteNeurale(tf.keras.Model):
    def __init__(self, dim_in,dim_out):
        super(ReteNeurale, self).__init__()
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(3,3), activation='tanh', input_shape=dim_in)  # Reduced filters, changed activation
        # Layer convoluzionali per estrarre caratteristiche
        self.pool1 = layers.MaxPooling2D((2, 2),data_format="channels_last")  # Riduce la dimensione dell'immagine
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2),data_format="channels_last")
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu',padding="same")
        self.pool3 = layers.MaxPooling2D((2, 2),data_format="channels_last")
        self.flatten = layers.Flatten(data_format="channels_last")  # Appiattisce (64, 64, 3) in (64*64*3)
        self.dropout= layers.Dropout(0.5) # Dropout per evitare overfitting
        self.dense2 = layers.Dense(128, activation='relu')
        self.policy = layers.Dense(dim_out, activation='softmax')
        self.value= layers.Dense(1)
    def call(self, inputs):
        #Controllo che l'input sia un tensore.
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)

        #input normalization
        inputs = tf.cast(inputs, tf.float32) / 255.0
        
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dropout(x)        
        x = self.dense2(x)
        policy=self.policy(x)
        value=self.value(x)
        return policy,value

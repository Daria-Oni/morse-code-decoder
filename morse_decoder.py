import random
import numpy as np
import string
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras import layers
import tensorflow as tf
from datetime import datetime


alphabet = list('abcdefghijklmnopqrstuvwxyz-')
values = ['.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..', '.---', '-.-', 
          '.-..', '--', '-.','---', '.--.', '--.-', 
          '.-.', '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..','-....-']

morse_dict = dict(zip(alphabet, values))
ascii_dict = dict(map(reversed, morse_dict.items())) # inverse mapping

# convert text to morse code
def morse_encode(text):
    t = ''.join([c for c in text.lower() if c in alphabet])
    return ' '.join([''.join(morse_dict[i]) for i in t])
 
# convert morse code to text
def morse_decode(code):
    return ''.join([ascii_dict[i] for i in code.split(' ')])

# -----------------------------------------------------------------------------
# generate data for training 
word_len = 6                                    # number of characters output
max_len_x = len(max(values, key=len))*word_len+(word_len-1)  # number of morse symbols
max_len_y = word_len
    
def generate_data(n):
    output_list = [(''.join([random.choice(string.ascii_lowercase + '-') 
        for _ in range(word_len)])) 
            for _ in range(n)]
    return output_list, [morse_encode(s) for s in output_list]
    
output_list, input_list = generate_data(10000)

# embedding, simply my mapping a one-hot encoding
class Embedding(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        
    def encode(self, token, num_rows):
        x = np.zeros((num_rows, len(self.chars)))   # zeros, except ..
        for i, c in enumerate(token):               # one 
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x):
        x = [x.argmax(axis=-1)]
        return ''.join(self.indices_char[int(v)] for v in x)

# -----------------------------------------------------------------------------
# prepare data for feeding to network
chars_in = '-. '
chars_out = ''.join(alphabet)

embedding_in = Embedding(chars_in)
embedding_out = Embedding(chars_out)

# x : input to encoder, y : output from decoder
x = np.zeros((len(input_list), max_len_x, len(chars_in)))
y = np.zeros((len(output_list), max_len_y, len(chars_out)))

for i, token in enumerate(input_list):
    x[i] = embedding_in.encode(token, max_len_x)
    
for i, token in enumerate(output_list):
    y[i] = embedding_out.encode(token, max_len_y)
    
# split data set : 3/4 for training, 1/4 for evaluation
m = 3*len(x)// 4
(x_train, x_val) = x[:m], x[m:]
(y_train, y_val) = y[:m], y[m:]

# -----------------------------------------------------------------------------
# experiment functions

def create_model_with_config(config):
    """Creates model with specified configuration"""
    model = Sequential()
    
    # first LSTM layer
    model.add(layers.LSTM(config['latent_dim'], 
                         input_shape=(max_len_x, len(chars_in))))
    
    if config.get('dropout', 0) > 0:
        model.add(layers.Dropout(config['dropout']))
    
    model.add(layers.RepeatVector(max_len_y))
    
    # second LSTM layer
    model.add(layers.LSTM(config['latent_dim'], return_sequences=True))
    
    if config.get('dropout', 0) > 0:
        model.add(layers.Dropout(config['dropout']))
        
    model.add(layers.TimeDistributed(layers.Dense(len(chars_out))))
    model.add(layers.Activation('softmax'))
    
    optimizer = 'adam'
    if config.get('learning_rate'):
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    model.compile(loss='categorical_crossentropy', 
                 optimizer=optimizer, 
                 metrics=['accuracy'])
    return model

def run_experiment(configs, experiment_name):
    """Run experiment with given configurations"""
    results = {}
    for name, config in configs.items():
        print(f"\nTraining {experiment_name}: {name}")
        start_time = datetime.now()
        
        model = create_model_with_config(config)
        history = model.fit(
            x_train, y_train,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # save results
        results[name] = {
            'history': history.history,
            'training_time': training_time,
            'final_val_acc': history.history['val_accuracy'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
        # test with sample cases
        test_cases = ['abcdef', '-hslu-', 'hahaha', '------', 'tttuuu']
        x_test = np.zeros((len(test_cases), max_len_x, len(chars_in)))
        for i, token in enumerate(test_cases):
            x_test[i] = embedding_in.encode(morse_encode(token), max_len_x)
        
        pred = model.predict(x_test)
        print(f"\nTest results for {name}:")
        for i in range(len(test_cases)):
            print(f"Input: {test_cases[i]} -> Output: {''.join([embedding_out.decode(code) for code in pred[i]])}")
    
    return results

def plot_experiment_results(results, title):
    """Plot comparison of results"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    for name, result in results.items():
        plt.plot(result['history']['val_accuracy'], label=name)
    plt.title(f'{title}: Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(122)
    for name, result in results.items():
        plt.plot(result['history']['val_loss'], label=name)
    plt.title(f'{title}: Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# define experiments

# test different architectures

architecture_configs = {
    'baseline': {
        'latent_dim': 200,
        'batch_size': 50,
        'epochs': 50
    },
    'very_small': {
        'latent_dim': 100,
        'batch_size': 50,
        'epochs': 50
    },
    'extra_large': {
        'latent_dim': 600,
        'batch_size': 50,
        'epochs': 50
    },
    'dropout_small': {
        'latent_dim': 200,
        'batch_size': 50,
        'epochs': 50,
        'dropout': 0.1
    },
    'dropout_large': {
        'latent_dim': 200,
        'batch_size': 50,
        'epochs': 50,
        'dropout': 0.3
    },
    'combined_large_dropout': {
        'latent_dim': 400,
        'batch_size': 50,
        'epochs': 50,
        'dropout': 0.2
    }
}

learning_configs = {
    'baseline': {
        'latent_dim': 200,
        'batch_size': 50,
        'epochs': 50,
        'learning_rate': 0.001
    },
    'tiny_batch': {
        'latent_dim': 200,
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 0.001
    },
    'large_batch': {
        'latent_dim': 200,
        'batch_size': 128,
        'epochs': 50,
        'learning_rate': 0.001
    },
    'very_low_lr': {
        'latent_dim': 200,
        'batch_size': 50,
        'epochs': 50,
        'learning_rate': 0.0001
    },
    'high_lr': {
        'latent_dim': 200,
        'batch_size': 50,
        'epochs': 50,
        'learning_rate': 0.01
    },
    'adaptive_batch': {
        'latent_dim': 200,
        'batch_size': 64,
        'epochs': 50,
        'learning_rate': 0.002
    }
}

# -----------------------------------------------------------------------------
# run experiments

print("Testing Network Architectures...")
arch_results = run_experiment(architecture_configs, "Architecture")
plot_experiment_results(arch_results, "Network Architecture")

print("\nArchitecture Results Summary:")
for name, result in arch_results.items():
    print(f"\n{name}:")
    print(f"Final validation accuracy: {result['final_val_acc']:.4f}")
    print(f"Training time: {result['training_time']:.1f} seconds")

print("\nTesting Learning Conditions...")
learning_results = run_experiment(learning_configs, "Learning Conditions")
plot_experiment_results(learning_results, "Learning Conditions")

print("\nLearning Conditions Results Summary:")
for name, result in learning_results.items():
    print(f"\n{name}:")
    print(f"Final validation accuracy: {result['final_val_acc']:.4f}")
    print(f"Training time: {result['training_time']:.1f} seconds")
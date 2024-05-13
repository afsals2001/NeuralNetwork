import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop



# Get the filepath of the input text file
input_file_path = tf.keras.utils.get_file('shakespeare.txt', 'input.txt')

# Read the text from the file and convert to lowercase
text = open(input_file_path, 'rb').read().decode(encoding='utf-8').lower()

# Get unique characters in the text
unique_characters = sorted(set(text))

# Create dictionaries to map characters to indices and vice versa
char_to_index = dict((char, i) for i, char in enumerate(unique_characters))
index_to_char = dict((i, char) for i, char in enumerate(unique_characters))

# Define sequence length and step size
SEQUENCE_LENGTH = 40
STEP_SIZE = 3

# Generate sequences and their corresponding next characters
sequences = []
next_characters = []

for i in range(0, len(text) - SEQUENCE_LENGTH, STEP_SIZE):
    sequences.append(text[i: i + SEQUENCE_LENGTH])
    next_characters.append(text[i + SEQUENCE_LENGTH])

# Initialize input and output data arrays
x = np.zeros((len(sequences), SEQUENCE_LENGTH, len(unique_characters)), dtype=np.bool_)
y = np.zeros((len(sequences), len(unique_characters)), dtype=np.bool_)

# Convert sequences and next characters into one-hot encoding
for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Define the model architecture
model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(unique_characters))))
model.add(Dense(len(unique_characters)))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# Train the model
model.fit(x, y, batch_size=256, epochs=4)

# Save the trained model
model.save("text_generator_model.keras")




#Step 2 - After generator the file you can use the below code to run and cmd the above code

# # Load the trained model
# model = tf.keras.models.load_model('text_generator_model.keras')

# # Define a function to sample from the predicted probabilities
# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)

# # Define a function to generate text
# def generate_text(length, temperature):
#     start_index = random.randint(0, len(text) - SEQUENCE_LENGTH - 1)
#     generated_text = ''
#     current_sequence = text[start_index: start_index + SEQUENCE_LENGTH]
#     generated_text += current_sequence
#     for i in range(length):
#         # Prepare input sequence for prediction
#         x = np.zeros((1, SEQUENCE_LENGTH, len(unique_characters)))
#         for t, char in enumerate(current_sequence):
#             x[0, t, char_to_index[char]] = 1

#         # Generate next character prediction
#         predictions = model.predict(x, verbose=0)[0]
#         next_index = sample(predictions, temperature)
#         next_character = index_to_char[next_index]

#         generated_text += next_character
#         current_sequence = current_sequence[1:] + next_character
#     return generated_text

# # Set the length and temperature for text generation
# TEXT_LENGTH = 300
# temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]

# # Generate text with different temperatures and print the results
# for temp in temperatures:
#     print('--------- Temperature:', temp, '---------')
#     print(generate_text(TEXT_LENGTH, temp))

    
          
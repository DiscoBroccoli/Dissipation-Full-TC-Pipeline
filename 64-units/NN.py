# import evaluation
from datasetTC.Production_prep import ProductionDataset
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers
from tensorflow import keras
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint

#%%

P = ProductionDataset()

#%%

x_train_standard = P._set_normalization_values(test=False)
x_test_standard = P._set_normalization_values(test=True)

#%%

y_train_shuffled = P.y_train_shuffled
y_test = P.y_test


#%%

def build_model(input_shape):
    """
    This function should build a Sequential model.
    The weights are initialised by providing the input_shape argument in the first layer.
    Your function should return the model.
    """
    initializer = tf.keras.initializers.HeUnifrom()
    regu = tf.keras.regularizers.l2(5e-4)

    model = tf.keras.Sequential([
        Dense(64, activation='relu', kernel_regularizer=regu, kernel_initializer=initializer, bias_initializer='zeros', input_shape=input_shape, name='dense_one'),
        BatchNormalization(),  # <- Batch normalisation layer
        Dense(64, activation='relu', kernel_regularizer=regu, kernel_initializer=initializer, bias_initializer='zeros', name='dense_two'),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=regu, kernel_initializer=initializer, bias_initializer='zeros', name='dense_three'),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_initializer=initializer, bias_initializer='zeros', name='dense_four'),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_initializer=initializer, bias_initializer='zeros', name='dense_five'),
        BatchNormalization(),
        Dense(1, kernel_initializer=initializer)
    ])

    return model


#%%

model = build_model(input_shape=(x_train_standard.shape[1],))
model.summary()

#%%

def get_weights(model):
    return [e.weights[0].numpy() for e in model.layers]

def get_biases(model):
    return [e.bias[0].numpy() for e in model.layers]

def plot_delta_weights(W0_layers):
    plt.figure(figsize=(8,8))
    delta_l = W0_layers[6]
    plt.imshow(delta_l)
    plt.title('Layer '+str(0))
    plt.axis('off')
    plt.colorbar()
    plt.suptitle('Weight matrices');

#%%

def compile_model(model):
    """
    This function takes in the model returned from the build_model function, and compiles it with an optimiser,
    loss function and metric.
    The function doesn't need to return anything; the model will be compiled in-place.
    """
    opt = keras.optimizers.Adam(learning_rate=5e-6)
    model.compile(loss=tf.keras.metrics.mean_squared_error, optimizer=opt,\
                  metrics=[tf.keras.metrics.MeanSquaredError(name='mse')])


compile_model(model)

#%%

def train_model(model, train_data, train_targets, epochs):
    """
    This function should train the model for the given number of epochs on the
    train_data and train_targets.
    Your function should return the training history, as returned by model.fit.
    """
    history = model.fit(train_data, train_targets, epochs=epochs, validation_split=0.2, batch_size=10, verbose=1,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])

    return history

#%%

history = train_model(model, x_train_standard, y_train_shuffled, epochs=2000)

#%%

example_batch = x_test_standard
example_result = model.predict(example_batch)
example_result = np.reshape(example_result, example_result.shape[0])
y_test = np.array(y_test)
y_test = np.reshape(y_test, y_test.shape[0])

# creating dictionnary
d = {'y/d': P.X_test['y/d'], 'pred_P': example_result, 'y_test': y_test}

fig = plt.gcf()
plt.figure
sns.set(rc={'figure.figsize':(12,10)})
ax = sns.lineplot(x = 'y/d', y = 'pred_P', marker="o", data = d)
ax = sns.lineplot(x = 'y/d', y = 'y_test', marker="o", data = d)
ax.set_title('Validation with test value')
plt.legend(['Predicted', 'Label'], loc='upper right')
plt.show()
fig.savefig('64_test.png', dpi=500)

#%%

# creating output file for test data
d = {'y/d': P.X_test['y/d'], 'pred_P': example_result, 'y_test': y_test, 'mse': np.square(example_result-y_test)}
output_df_test = pd.DataFrame(data=d).sort_values(by=['y/d'])
output_df_test.to_excel("64_output_df_test.xlsx")
#%%
epoch = len(history.history['loss'])
list_epoch = [ x+1 if x % 1 == 0 else x for x in range(epoch)]

d = {'epochs': [ x+1 if x % 1 == 0 else x for x in range(epoch)], 'loss': history.history['loss'],
     'val_loss': history.history['val_loss']}

output_df_loss = pd.DataFrame(data=d).sort_values(by=['epochs'])
output_df_loss.to_excel("64_output_df_loss.xlsx")

#%%

validate_batch = P._set_normalization_values_validation(test=False)
validate_batch = model.predict(validate_batch[:77])
validate_batch = np.reshape(validate_batch, validate_batch.shape[0])
y_train = np.array(P.y_train[:77])
y_train = np.reshape(y_train, y_train.shape[0])

# creating dictionnary
d = {'y/d': P.X_train['y/d'][:77], 'pred_P': validate_batch, 'y_train': y_train}
output_df_validation = pd.DataFrame(data=d).sort_values(by=['y/d'])
fig = plt.gcf()
plt.figure
sns.set(rc={'figure.figsize':(12,10)})
ax = sns.lineplot(x = 'y/d', y ='pred_P', marker="o", data = d)
ax = sns.lineplot(x = 'y/d', y = 'y_train', data = d)
plt.legend(['Training', 'Validation'], loc='upper right')
ax.set_title('Validation with train value')

plt.show()
fig.savefig('64_validation.png', dpi=500)

#%%
# creating output file for validation data
d = {'y/d': P.X_train['y/d'][:77], 'pred_P': validate_batch, 'y_train': y_train, 'mse': np.square(validate_batch-y_train)}
output_df_validation = pd.DataFrame(data=d).sort_values(by=['y/d'])
output_df_validation.to_excel("64_output_df_validation.xlsx")

W_layers = get_weights(model)

print(W_layers[4])
# weight_output = sess.run([W_layers])

# np.savetxt("weight_output.csv", W_layers)

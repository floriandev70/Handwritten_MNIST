import tensorflow as tf
import keras, numpy
###################################################
# DEFINE PROBLEM #
#--------------------------------------------------
# Erkenne hangeschriebene Zahlen aus MNIST
# Jede Zahl hat 28x28 sw Pixel
def handwritten():
    ###################################################
    # IMPORT DATA #
    #--------------------------------------------------
    # Daten laden
    my_mnist = keras.datasets.mnist.load_data()
    # Daten unpacken
    (x_train, y_train), (x_test, y_test) = my_mnist

    ###################################################
    # PREPARE DATA #
    #--------------------------------------------------
    # Wie sehen die Daten aus?
    # Erstma Ausgabeformat anpassen
    numpy.set_printoptions(linewidth=120)
    # Shapes
    print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    # Daten via Tabellenview ansehen hier Breakpoint setzen
    # Pixelfarbe auf Intervall [0..1] konvertieren = x_train/255 und x_test/255
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print('konvertiert')

    ###################################################
    # CREATE MODEL #
    #--------------------------------------------------
    # 28x28 float inputs
    # 1x float output
    # flodis example model definition
    my_model = tf.keras.models.Sequential([
      # Input layer
      tf.keras.layers.Flatten(input_shape=(28, 28), name='input_layer'),
      # Hidden layer
      tf.keras.layers.Dense(128, activation='relu', name='hidden_layer'),
      # Dropout layer
      tf.keras.layers.Dropout(0.2, name='dropout_layer'),
      # Output layer
      tf.keras.layers.Dense(10, name='output_layer')
    ], name = 'my_model')
    # model overview
    #print(my_model.summary())

    ###################################################
    # INVESTIGATE MODEL #
    #--------------------------------------------------
    # Aktuelle Prediction für das erste Element:
    numpy.set_printoptions(linewidth=400)
    print(x_train[:1].shape)
    print(x_train[0].shape)
    # Was käme für das erste Element raus wenn man das Modell untrainiert befragt?
    my_y = my_model(x_test[:1]).numpy()
    print('Input (Buchstabe) = ', y_test[0])
    print('Prediction = ', my_y)
    # In Wahrscheinlichkeiten umwandeln
    my_y_prob = tf.nn.softmax(my_y).numpy()
    print('Wahrscheinlichkeiten = ', my_y_prob)
    print('Voraussage = ', my_y_prob.argmax(axis=-1)[0]+1)
    predictions = my_model(x_test[:]).numpy()
    predictions = tf.nn.softmax(predictions).numpy()
    predictions = predictions.argmax(axis=-1)[:]+1
    print('Predictions = ', predictions)
    match = 0
    for i in range(predictions.shape[0]):
      if predictions[i] == y_test[i]:
        #print(predictions[i])
        match += 1
    percentage = match/predictions.shape[0]*100
    print('Matches = ', match, 'von ', predictions.shape[0], 'das sind ', f"{percentage:10.2f}", '%')

    ###################################################
    # TRAIN MODEL #
    #--------------------------------------------------
    # loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # compile model
    my_model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    # training
    my_model.fit(x_train, y_train, epochs=7)
    ###################################################
    # CREATE OUTPUT #
    #--------------------------------------------------
    # evaluiere Modellperformanze
    my_model.evaluate(x_test,  y_test, verbose=2)

    ###################################################
    # ANSWER PROBLEM #
    #--------------------------------------------------
    predictions = my_model(x_test[:]).numpy()
    predictions = tf.nn.softmax(predictions).numpy()
    predictions = predictions.argmax(axis=-1)[:]
    print('Predictions = ', predictions)
    match = 0
    for i in range(predictions.shape[0]):
      if predictions[i] == y_test[i]:
        #print('match: ', predictions[i])
        match += 1
      else:
        print('no match: ', predictions[i], '<>', y_test[i])
    percentage = match/predictions.shape[0]*100
    print('Matches = ', match, 'von ', predictions.shape[0], 'das sind ', f"{percentage:10.2f}", '%')

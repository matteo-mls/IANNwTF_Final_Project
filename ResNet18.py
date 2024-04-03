import tensorflow as tf

# - IdentityBlock --------------------------------------------------------------------------
class IdentityBlock(tf.keras.layers.Layer):
  # Constructor
  def __init__(self, filters):
    super(IdentityBlock, self).__init__()
    # Define the Architecture of the IdentityBlock
    self.layerList = [
      # Layer 1
      tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = "same", strides= (1, 1)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Activation("relu"),
      # Layer 2
      tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = "same", strides = (1, 1)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3)
    ]


  @tf.function
  def call(self, x, training = False):
    xSkip = x  # save the input in a variable called xSkip
    for layer in self.layerList:
      # both Batchnormalization and Dropout Layer act differently depending on the training Flag
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        x = layer(x, training)
      elif isinstance(layer, tf.keras.layers.Dropout):
        x = layer(x, training)
      else:
        x = layer(x)
    x = x + xSkip  # add the Residue xSkip
    x = tf.nn.relu(x)
    return x

# - ProjectionBlock --------------------------------------------------------------------
class ProjectionBlock(tf.keras.layers.Layer):
  # Constructor
  def __init__(self, filters):
    super(ProjectionBlock, self).__init__()
    # Define the Architecture of the ProjectionBlock
    self.layerList = [
      # Layer 1
      tf.keras.layers.Conv2D(filters, kernel_size = (3, 3), padding = "same", strides = (2, 2)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Activation("relu"),
      # Layer 2
      tf.keras.layers.Conv2D(filters, kernel_size = (3, 3), padding = "same", strides = (1, 1)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3)
    ]
    self.convo = tf.keras.layers.Conv2D(filters, kernel_size = (1, 1), padding = "same", strides = (2, 2))  # to align the different dimensionalities


  @tf.function
  def call(self, x, training = False):
    xSkip = x  # save the input in a variable called xSkip
    for layer in self.layerList:
      # both Batchnormalization and Dropout Layer act differently depending on the training Flag
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        x = layer(x, training)
      elif isinstance(layer, tf.keras.layers.Dropout):
        x = layer(x, training)
      else:
        x = layer(x)
    xSkip = self.convo(xSkip)  # align the different dimensionalities
    x = x + xSkip  # add the Residue xSkip
    x = tf.nn.relu(x)
    return x

# - ResNet 18 -----------------------------------------------------------------------------
class ResNet18Classifier(tf.keras.Model):
  # Constructor
  def __init__(self, output_size, lr):
    super(ResNet18Classifier, self).__init__()   # call the constructor of the super class

    # - The Architecture ------------------------------------------------------------------
    # initialize all Layers and save them as class attributes
    # first two Layers are the same in every ResNet Architecture, a conv2D Layer and a MaxPooling Layer
    self.firstConvLayerList = [  # input shape is (batch_size, 64, 64, 3)
    # activation = None means linear Activation (because we do apply relu after Batchnormalization)
      tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = "same", activation = None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2) , padding = "same")
    ]

    # the Layers in between the first two Layers and the last two Layers are different in the various ResNet Architectures
    # here a ResNet18 Architecture is build [2, 2, 2, 2]
    self.layersInBetween = [
      # here the dimensionalities still match, so that no ProjectionBlock is needed in the first ResNet Block
      IdentityBlock(filters = 64),
      IdentityBlock(filters = 64),
      # second ResNet Block, the dimensionalities mismatch and therefore a ProjectionBlock is needed
      ProjectionBlock(filters = 128),
      IdentityBlock(filters = 128),
      # third ResNet Block, the dimensionalities mismatch and therefore a ProjectionBlock is needed
      ProjectionBlock(filters = 256),
      IdentityBlock(filters = 256),
      # fourth ResNet Block, the dimensionalities mismatch and therefore a ProjectionBlock is needed
      ProjectionBlock(filters = 512),
      IdentityBlock(filters = 512),
    ]

    # the last Dense Layers are also the same in every ResNet
    self.lastDenseLayerList = [
      tf.keras.layers.GlobalAveragePooling2D(), # Anzahl Featuremaps viele Werte pro Batch
      tf.keras.layers.Dense(10, activation = "softmax")
    ]

    # - Optimizer and Loss ---------------------------------------------------------------
    # initialize and save the Optimizer and the Loss Function Object as class attributes
    self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = False)

    # - Metrics --------------------------------------------------------------------------
    # initialize multiple Metrics and save them as class variables, the keras.Model will make them available in a List named self.metrics
    self.metric_loss = tf.keras.metrics.Mean(name = "loss")  # Metric for the Calculation of the Loss
    self.metric_accuracy = tf.keras.metrics.Accuracy(name = "accuracy")  # Metric for the Calculation of the Accuracy


  @tf.function
  def call(self, x, training = False):  # - Feedforward Function -------------------------------------------
    # each layer gets the output of the previous layer as input
    for layer in self.firstConvLayerList:
      # Batchnormalization acts differently depending on the training Flag
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        x = layer(x, training)
      else:
        x = layer(x)
    for layer in self.layersInBetween:
      x = layer(x, training)
    for layer in self.lastDenseLayerList:
      x = layer(x)
    return x


  @tf.function
  def train_step(self, x, target):  # - Training Function ---------------------------------
    # open the Gradient Tape and calculate the Feedforward Step and the Loss
    with tf.GradientTape() as tape:
      prediction = self(x, training = True)  # set the training Flag to True
      loss = self.loss_function(target, prediction)
    # use the Gradient Tape to calculate the Gradients (Backpropagation)
    gradients = tape.gradient(loss, self.trainable_variables)
    # use the Optimizer to update the weights based on the calculated Gradients
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.metric_loss.update_state(loss)  # update the Loss Metric
    # get the current prediction and update the Accuracy Metric
    prediction = tf.argmax(prediction, axis = -1)
    label = tf.argmax(target, axis = -1)
    self.metric_accuracy.update_state(label, prediction) # update the Accuracy Metric


  @tf.function
  def test_step(self, dataset):  # - Testing Function ---------------------------------------
    # Reset all Metrics: Do not mix Training and Testing Metrices together!!!
    self.metric_loss.reset_states()
    self.metric_accuracy.reset_states()
    for x, target in dataset:  # iterate over the test dataset
      prediction = self(x, training = False)
      # get the current Loss and update the Loss Metric
      loss = self.loss_function(target, prediction)
      self.metric_loss.update_state(loss)  # update the Loss Metric
      # get the current prediction and update the Accuracy Metric
      prediction = tf.argmax(prediction, axis = -1)
      label = tf.argmax(target, axis = -1)
      self.metric_accuracy.update_state(label, prediction)  # update the Accuracy Metric


  # this method runs in eager mode and not in graph mode
  def confuTest(self, dataset):
    ground = []
    pred = []
    for x, target in dataset:  # iterate over the test dataset
      prediction = self(x, training = False)
      prediction = tf.argmax(prediction, axis = -1)  # get prediction
      label = tf.argmax(target, axis = -1)  # get ground truth
      # append prediction and ground truth to the lists
      ground.append(label)
      pred.append(prediction)

    return ground, pred
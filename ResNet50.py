import tensorflow as tf

# - ConvolutionalBlock --------------------------------------------------------------------
class base_conv_block(tf.keras.layers.Layer):
  # Constructor
  def __init__(self, filters, kernel_size, strides, padding = "same"):
    super(base_conv_block, self).__init__()
    # Define the Architecture of the ConvolutionalBlock
    self.layerList = [
      tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Activation("relu")
    ]


  @tf.function
  def call(self, x, training = False):
    for layer in self.layerList:
      # both Batchnormalization and Dropout Layer act differently depending on the training Flag
      if isinstance(layer,tf.keras.layers.BatchNormalization):
        x = layer(x, training)
      elif isinstance(layer, tf.keras.layers.Dropout):
        x = layer(x, training)
      else:
        x = layer(x)
    return x

# - IdentityBlock --------------------------------------------------------------------
class Identity_Block(tf.keras.layers.Layer):
  # Constructor
  def __init__(self, filters):
    super(Identity_Block, self).__init__()
    # Define the Architecture of the IdentityBlock
    self.layerList = [
      base_conv_block(filters = filters, kernel_size = (1, 1), strides = (1, 1)),
      base_conv_block(filters = filters, kernel_size = (3, 3), strides = (1, 1)),
      tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3)
    ]


  @tf.function
  def call(self, x, training = False):
    xSkip = x
    for layer in self.layerList:
      # both Batchnormalization and Dropout Layer act differently depending on the training Flag
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        x = layer(x, training)
      elif isinstance(layer, tf.keras.layers.Dropout):
        x = layer(x, training)
      else:
        x = layer(x)
    x = x + xSkip
    x = tf.nn.relu(x)
    return x

# - ProjectionBlock --------------------------------------------------------------------
# Projection Blocks are used when the dimension of input and output is not the same
class Projection_Block(tf.keras.layers.Layer):
  # Constructor
  def __init__(self, filters, strides):
    super(Projection_Block, self).__init__()
    # Define the Architecture of the ProjectionBlock
    # first step
    self.layerList = [
      base_conv_block(filters = filters, kernel_size = (1, 1), strides = strides),
      base_conv_block(filters = filters, kernel_size = (3, 3), strides = (1, 1)),
      tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1)),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3)
    ]
    # second step, for residual connection
    self.layerListRes = [
      tf.keras.layers.Conv2D(filters = filters * 4, kernel_size = (1, 1), strides = strides),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.3)
    ]


  @tf.function
  def call(self, x, training = False):
    xSkip = x
    for layer in self.layerList:
      # both Batchnormalization and Dropout Layer act differently depending on the training Flag
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        x = layer(x, training)
      elif isinstance(layer, tf.keras.layers.Dropout):
        x = layer(x, training)
      else:
        x = layer(x)
    for layerr in self.layerListRes:
      # both Batchnormalization and Dropout Layer act differently depending on the training Flag
      if isinstance(layerr, tf.keras.layers.BatchNormalization):
        xSkip = layerr(xSkip, training)
      elif isinstance(layerr,tf.keras.layers.Dropout):
        xSkip = layerr(xSkip, training)
      else:
        xSkip = layerr(xSkip)
    x = x + xSkip
    x = tf.nn.relu(x)
    return x

# - ResNet 50 -----------------------------------------------------------------------------
class ResNet50Classifier(tf.keras.Model):
  # Constructor
  def __init__(self ,output_size, lr):
    super(ResNet50Classifier, self).__init__()

    # - The Architecture ------------------------------------------------------------------
    # initialize all Layers and save them as class attributes
    # first two Layers are the same in every ResNet Architecture, a conv2D Layer and a MaxPooling Layer
    self.fistConvLayerList = [
      tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), padding = "same", activation = None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = "same")
    ]

    # the Layers in between the first two Layers and the last two Layers are different in the various ResNet Architectures
    # here a resnet50 Architecture is build [3, 4, 6, 3] which is the same structure as 34, but the difference is inside the blocks,
    # instead of two layes here they have three [1, 3, 1], where the last layer has a four times the filter of the previous
    self.layersInBetween = [
      # fist convolutional block
      Projection_Block(filters = 64, strides = (1, 1)),
      Identity_Block(filters = 64),
      Identity_Block(filters = 64),
      # second convolutional block
      Projection_Block(filters = 128, strides = (2, 2)),
      Identity_Block(filters = 128),
      Identity_Block(filters = 128),
      Identity_Block(filters = 128),
      # third convolutional block
      Projection_Block(filters = 256, strides = (2, 2)),
      Identity_Block(filters = 256),
      Identity_Block(filters = 256),
      Identity_Block(filters = 256),
      Identity_Block(filters = 256),
      Identity_Block(filters = 256),
      # fourth convolutional block
      Projection_Block(filters = 512, strides = (1, 1)),
      Identity_Block(filters = 512),
      Identity_Block(filters = 512),
    ]

    # the last Dense Layers are also the same in every ResNet
    self.lastDenseLayerList = [
      tf.keras.layers.GlobalAveragePooling2D(),
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
  def call(self, x, training = False):  # - Feedforward Function --------------------------
  # each layer gets the output of the previous layer as input
    for layer in self.fistConvLayerList:
      # Batchnormalization acts differently depending on the training Flag
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        x = layer(x, training)
      else:
        x = layer(x)
    for layer in self.layersInBetween:
      x = layer(x, training)
    for layer in self.lastDenseLayerList:
      x = layer(x)  # no batchnorm
    return x


  @tf.function
  def train_step(self, x, target):  # - Training Function ---------------------------------
  # open the Gradient Tape and calculate the Feedforward Step and the Loss
    with tf.GradientTape() as tape:
      prediction = self(x, training = True) # set the training flag to True
      loss = self.loss_function(target, prediction)
    # use the Gradient Tape to calculate the Gradients (Backpropagation)
    gradients = tape.gradient(loss, self.trainable_variables)
    # use the Optimizer to update the wieghts based on the calculated Gradients
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    self.metric_loss.update_state(loss) # update the Loss Metric
    # get the current prediction and update the Accuracy Metric
    prediction = tf.argmax(prediction, axis = -1)
    label = tf.argmax(target, axis = -1)
    self.metric_accuracy.update_state(label,prediction) # update the Accuracy Metric


  @tf.function
  def test_step(self, dataset):  # - Testing Function -------------------------------------
    # Reset all Metrics: Do not mix Training and Testing Metrices together!!!
    self.metric_loss.reset_state()
    self.metric_accuracy.reset_state()
    for x, target in dataset: # iterate over the test dataset
      prediction = self(x, training = False) # set the training flag to False
      # get the current Loss and update the Loss Metric
      loss = self.loss_function(target, prediction)
      self.metric_loss.update_state(loss) # update the Loss Metric
      # get the current prediction and update the Accuracy Metric
      prediction = tf.argmax(prediction, axis = -1)
      label = tf.argmax(target, axis = -1)
      self.metric_accuracy.update_state(label,prediction) # update the Accuracy Metric


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
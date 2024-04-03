import tensorflow as tf
import tqdm as tqdm
import os
import datetime

from ResNet18 import *
from ResNet34 import *
from ResNet50 import *
from preprocessing import *
from utils import *


PATH = save_path()
EUROPATH = PATH + os.sep + "eurosat"

# Dataset Parameters
test_size = 0.3  # size of the test dataset
threshold = 2000  # how many samples per class do we want? (augmented + original)
da = 'base'  # parameter for data augumentation, base

# Hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 0.0001

# choose the Model, ResNet18, ResNet34 or ResNet50
MODEL = ResNet18Classifier(output_size = 10, lr = LEARNING_RATE)
#MODEL = ResNet34Classifier(output_size = 10, lr = LEARNING_RATE)
#MODEL = ResNet50Classifier(output_size = 10, lr = LEARNING_RATE)


def main():
  # - Load Dataset ------------------------------
  train_ds, test_ds, info = loadData(test_size, threshold, da)
  visualizeDataSet(train_ds, test_ds)

  # - Apply Preprocessing Functions -------------
  train_dataset = train_ds.apply(prepare_data)
  test_dataset = test_ds.apply(prepare_data)

  # - Logging -----------------------------------
  current_time = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
  file_path = EUROPATH + os.sep + "Logs" + os.sep + current_time
  train_summary_writer = tf.summary.create_file_writer(file_path)

  # - Initialize Model --------------------------
  classifier = MODEL
  classifier.build(input_shape = (None, 64, 64, 3))  # batch_size is None for this test
  classifier.summary()

  # - Train and test loss/accuracy --------------
  print(f"Epoch 0")  # log initial accuracy and loss
  log(train_summary_writer, classifier, train_dataset, test_dataset, 0)

  # - Train loop --------------------------------
  for epoch in range(1, NUM_EPOCHS + 1):

    print(f"Epoch {epoch}")

    for x, target in tqdm.tqdm(train_dataset, position = 0, leave = True):
      classifier.train_step(x, target)

    # log the results of this epoch and reset metrics, this method calls the test_step function
    log(train_summary_writer, classifier, train_dataset, test_dataset, epoch)

    # Save model (its parameters)
    # classifier.save_weights(EUROPATH + os.sep + "Models" + os.sep + current_time + f"_trained_weights_{epoch}", save_format = "tf")
  

  plot(EUROPATH)
  # - Confusion ---------------------------------
  # After Training is complete, calculate the confusion Matrix
  ground, pred = classifier.confuTest(test_dataset)  # let the model classify all samples in the test dataset
  classNames = info.features["label"].names  # extract the class names out of the dataset info
  plotPath = EUROPATH + os.sep + "Plots" + os.sep + current_time
  confu(classNames, ground, pred, plotPath)  # create confusion matrix based on the ground truth values and the models predictions


# Let"s GOOOOOO!
if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    print("KeyboardInterrupt received")
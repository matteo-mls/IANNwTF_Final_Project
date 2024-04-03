import os
import tensorflow as tf
from tbparse import SummaryReader
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


# - Create Directory Structure ------------------------------------------------
def save_path():
    # Define the base path
    PATH = os.path.dirname(os.path.realpath(__file__))
    
    if os.path.exists(PATH):  # Check if this directory exists
        if not os.path.exists(os.path.join(PATH, "eurosat")):
            os.chdir(PATH)
            os.mkdir("eurosat")
            os.chdir(os.path.join(PATH, "eurosat"))
            os.mkdir("Logs")
            os.mkdir("Models")
            os.mkdir("Plots")
            print("Created Directory Structure!")
        else:
            print("Directory Structure already exists :), let's go!")
    else:
        print("Invalid Path provided :( ...)")
    
    return PATH

# - Load Logs as a Dataframe --------------------------------------------------
def load_dataframe(log_dir):
  reader = SummaryReader(log_dir)  # create a SummaryReader
  df = reader.tensors  # read the saved tensors

  # rename and set the Index of the tensor dataframe
  df = df.rename(columns={"step": "Epoch"})
  df = df.set_index(["Epoch"])

  # for each tag, there must be a column
  tags = df.loc[:, "tag"].unique()

  data = {}  # empty Dictionary

  for tag in tags:  # iterate over all tags
    mask = df["tag"] == tag
    df_tmp = df.loc[mask]
    new_tag = tag.replace("_", " ")
    data[new_tag] = df_tmp.value

  # convert data into pandas Dataframe and return it
  df = pd.DataFrame(data)
  return df

# - Plot the loaded Dataframe --------------------------------------------------
def plot(path):  # path should equal EUROPATH
  log_dir = path + os.sep + "Logs"

  df = load_dataframe(log_dir)  # load the log as a panda
  print(df)

  # define a plot with two subplots
  fig, axes = plt.subplots(1, 2)

  # Accuracy Subplot
  sns.lineplot(data=df.loc[:, ["train accuracy", "test accuracy"]], ax = axes[0], markers = True)
  axes[0].set_title("Accuracy")

  # Loss Subplot
  sns.lineplot(data=df.loc[:, ["train loss", "test loss"]], ax = axes[1], markers = True)
  axes[1].set_title("Loss")

  # grid
  for ax in axes.flatten():
    ax.grid()

  plt.tight_layout()
  plt.savefig(path + os.sep + "Plots" + os.sep + "AccuracyLoss.png")  # save the plot
  plt.show()

# - Visualization of the amount of samples per class ----------------------------
def visualizeDataSet(train, test):
  sns.set()  #set seaborn plotting aesthetics as default
  figure, axes = plt.subplots(1, 2, sharex = True, figsize = (13, 5))  # define the plotting region: 1 row, 3 columns
  figure.suptitle("Amount of Samples per Class for every Subdataset")  # Title of the Figure
  # define Subtitles for every Subplot
  axes[0].set_title("Train Dataset")
  axes[1].set_title("Test Dataset")
  # Set the x Label for each Subplot
  axes[0].set_xlabel("Classes")
  axes[1].set_xlabel("Classes")
  # Set the y Label for each Subplot
  axes[0].set_ylabel("Number of Samples per Class")
  axes[1].set_ylabel("Number of Samples per Class")

  # count how many Samples per Class are in each Subset
  labels, counts = np.unique(np.fromiter(train.map(lambda img, label: label), np.int32), return_counts = True)
  sns.barplot(x = labels, y = counts, ax = axes[0])
  labels, counts = np.unique(np.fromiter(test.map(lambda img, label: label), np.int32), return_counts = True)
  sns.barplot(x = labels, y = counts, ax = axes[1])
    
  figure.show()

# - Confusion Matrix ------------------------------------------
def confu(classNames, ground: tf.TensorArray, pred: tf.TensorArray, plotPath: str):
  # - From Tensors to 1D Lists --------------------------------
  # Note: each List is 2D and the shape depends on the batchsize (is either a multiple of the batchsize or the rest)!
  groundList = []
  for i in range(len(ground)):
    for j in range(len(ground[i])):
      tensor = ground[i][j].numpy()
      groundList.append(tensor)
  # print("ground truth", groundList)

  predList = []
  for i in range(len(pred)):
    for j in range(len(pred[i])):
      tensor = pred[i][j].numpy()
      predList.append(tensor)
  # print("predictions", predList)

  # - Confusion Matrix -----------------------------------------
  # note: if we use ground first and pred as second argument, then y label of the plot is ground, while x label of plot is pred
  # but i prefer it the other way around, which is why pred is used first so y is pred and x is ground
  # if we swap these two, then the calculations below have to be changed as well!!!
  conMatrix = confusion_matrix(predList, groundList)
  # print(conMatrix)

  # - Diagonal, Sum over each Row & Sum over each Column -------
  # calculate sums of each column as well as sums of each row and extract the diagonal of the matrix
  colSum = conMatrix.sum(axis = 0)  # sum of each column, this contains the amount of ground truth samples for each class
  rowSum = conMatrix.sum(axis = 1)  # sum of each row
  diagonal = np.diag(conMatrix)  # amount of correct classified samples per class
  # print("diagonal", diagonal)
  # print("colSum", colSum)
  # print("rowSum", rowSum)

  # - OA, UA and PA ---------------------------------------------
  # calculate producer, user and overall accuracy
  #producerAcc = diagonal / colSum
  producerAcc = np.divide(diagonal, colSum, out = np.zeros(diagonal.shape, dtype = float), where = colSum != 0)  # careful with division by zero
  #userAcc =  diagonal / rowSum
  userAcc = np.divide(diagonal, rowSum, out = np.zeros(diagonal.shape, dtype = float), where = rowSum != 0)  # careful with division by zero
  overallAcc = diagonal.sum(axis = 0) / colSum.sum(axis = 0) * 100
  print("\nOverall Accuracy: ", overallAcc)
  print("\nUser Accuracy:", userAcc)
  # print("\nProducer Accuracy:", producerAcc)

  # - Confidence Interval for OA ---------------------------------
  # calculate Confidence Interval for the Overall Accuracy (OA)
  xOA = diagonal.sum(axis = 0)  # amount of correct classified samples
  nOA = np.nansum(rowSum)  # amount of ground truth samples (for the Overall Accuracy it is the total amount of samples)
  pOA = xOA / nOA  # Overall Accuracy
  zScore = 1.96  # 95 % CI, Fixed value taken from table (5 % unconfidence)
  CI_OA_low = pOA - (zScore / nOA) * (np.sqrt((xOA * (nOA - xOA)) / (nOA + zScore - 1)))  # lower Bound
  CI_OA_up = pOA + (zScore / nOA) * (np.sqrt((xOA * (nOA - xOA)) / (nOA + zScore - 1)))  # upper Bound
  print(CI_OA_low, CI_OA_up)

  # - Confidence Intervals for UA --------------------------------
  # calculate Confidence Intervals for the User Accuracies (UA)s
  n = rowSum  # total amount of ground truth samples per class
  x = diagonal  # amount of correct classified samples per class
  #p = x / n
  p = np.divide(x, n, out = np.zeros(x.shape, dtype = float), where = n != 0)  # careful with division by zero
  CI_low = []  # lower Bounds
  CI_up = []  # upper Bounds

  for i in range(len(n)):
    if n[i] == 0:
      frac1 = 0
    else:
      frac1 = (zScore / n[i])

    if (n[i] + zScore - 1) == 0:
      frac2 = 0
    else:
      frac2 = (x[i] * (n[i] - x[i])) / (n[i] + zScore - 1)

    help = frac1 * np.sqrt(frac2)
    CI_low.append(p[i] - help)
    CI_up.append(p[i] + help)

  print()
  print("Lower Bounds:\n", CI_low)
  print()
  print("Upper Bounds:\n", CI_up)

  # - Visualize Confusion Matrix --------------------------------
  plt.figure(figsize = [7, 6])
  ax = plt.gca()
  plt.imshow(conMatrix, cmap = plt.cm.Blues)
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position("top")
  plt.colorbar(fraction = 0.04)
  plt.xticks(np.arange(len(classNames)), classNames)
  plt.yticks(np.arange(len(classNames)), classNames)
  plt.ylabel("Prediction", labelpad = 13)
  plt.xlabel("Ground Truth", labelpad = 13)
  plt.setp(ax.get_xticklabels(), rotation = 90, horizontalalignment = "center")
  for i in range(conMatrix.shape[0]):
    for j in range(conMatrix.shape[1]):
      plt.text(j, i, conMatrix[i, j], horizontalalignment = "center", color = "white" if conMatrix[i, j] > (conMatrix.max() / 2.0) else "black")
  plt.savefig(plotPath, bbox_inches = "tight")  # save the plot
  plt.show()

# - Log Train and Test Losses and Accuracies ---------------------
def log(train_summary_writer, classifier, train_dataset, test_dataset, epoch):
  # Epoch 0 = no training steps are performed
  # test based on train data
  # -> Determinate initial train_loss and train_accuracy
  if epoch == 0:
    classifier.test_step(train_dataset.take(5000))

  # - Train ------------------------------------------------------
  # get the current Training results
  train_loss = classifier.metric_loss.result()
  train_accuracy = classifier.metric_accuracy.result()
  # reset training metrics
  classifier.metric_loss.reset_states()
  classifier.metric_accuracy.reset_states()

  # - Test -------------------------------------------------------
  classifier.test_step(test_dataset)
  # get the current Test results
  test_loss = classifier.metric_loss.result()
  test_accuracy = classifier.metric_accuracy.result()
  # reset test metrics
  classifier.metric_loss.reset_states()
  classifier.metric_accuracy.reset_states()

  # - Write to TensorBoard ---------------------------------------
  with train_summary_writer.as_default():
    tf.summary.scalar(f"train_loss", train_loss, step=epoch)
    tf.summary.scalar(f"train_accuracy", train_accuracy, step=epoch)

    tf.summary.scalar(f"test_loss", test_loss, step=epoch)
    tf.summary.scalar(f"test_accuracy", test_accuracy, step=epoch)

  # - Output -----------------------------------------------------
  print(f"train_loss:     {train_loss}")
  print(f"test_loss:      {test_loss}")
  print(f"train_accuracy: {train_accuracy}")
  print(f"test_accuracy:  {test_accuracy}")

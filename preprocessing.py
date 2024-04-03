import tensorflow_datasets as tfds
import tensorflow as tf
import random
import numpy as np

# - Data Loading Pipeline -------------------------------------------------
def loadData(test_size, threshold, da):
  # Load the whole dataset
  train_ds, info = tfds.load("eurosat", split = "train", as_supervised = True, with_info = True)

  # check which scenario is requested
  if threshold > 2000:
    threshold = 2000
  if da != "base":
    threshold = 2000

  train_data = list(train_ds)  # convert datasets to lists for easier manipulation
  random.shuffle(train_data)  # shuffle to not add biases

  # get 2000 samples for each class
  filtered_data = []
  for i in range(10):
      temp_data = []  # create a temporary list for each iteration
      for img, label in train_data:
          if i == label and len(temp_data) < 2000:  # 2000 is the threshold, label 5 does not have more than 2000 samples
              temp_data.append((img, label))
      filtered_data += temp_data  # we add the data, they are kind of sorted by label, but then they will be splitted and shuffled later
      temp_data = []

  """
  # prints how many samples per label
  for i in range(10):
    c = 0
    for img, label in filtered_data:
        if label == i:
          c += 1
    print("Label", i, " Samples: ", c)
  """

  # get the same amount of samples for each class in training and test
  train_dat = []
  test_dat = []
  partial_list = []

  if threshold == 2000:
    threshold = int(threshold * (1 - test_size))

  n = 0
  m = int(len(filtered_data) // 10)  # 2000
  train_index = int(threshold)  # e.g., 100, 500, 1000
  test_idx = int((len(filtered_data) // 10) * test_size) # 2000 * test_size = if test_size = 0.2 -> 400
  test_index = train_index+test_idx

  for i in range(10):
    partial_list = filtered_data[n:m]  # each iteration get the segment of the list with the same label
    train_dat = train_dat + partial_list[:train_index]
    test_dat = test_dat + partial_list[train_index:test_index]
    partial_list = []  # empty the list for the next iteration
    n = m  # move the beginning of the segment of one step (where previously was the end)
    m = m + 2000  # move the end of the segment of one step (the threshold)

  print("Training samples:", len(train_dat))
  print("Test samples:", len(test_dat))

  if da == "base":
    # convert the data into TensorFlow datasets with proper data types as from the original tf.dataset
    train_ds = tf.data.Dataset.from_tensor_slices((np.array([x[0] for x in train_dat], dtype = np.uint8), np.array([x[1] for x in train_dat], dtype = np.int64)))
    test_ds = tf.data.Dataset.from_tensor_slices((np.array([x[0] for x in test_dat], dtype = np.uint8), np.array([x[1] for x in test_dat], dtype = np.int64)))

  elif isinstance(da, int):
    original_size = da  # amount of training samples per class
    random.shuffle(train_dat)  # shuffle the list not to bias the picking of samples

    # initialize a dictionary to store samples for each label
    samples_per_label = {label: [] for label in range(10)}
    samples_count_per_label = {label: 0 for label in range(10)}

    # iterate over the dataset and collect samples for each label
    for img, label in train_dat:
      label_int = label.numpy()  # convert TensorFlow tensor to integer
      if samples_count_per_label[label_int] < original_size:  # da samples per class for instance 100-500-1000
        samples_per_label[label_int].append((img, label_int))
        samples_count_per_label[label_int] += 1

    # convert the dictionary of lists to a single list
    selected_samples = [sample for sublist in samples_per_label.values() for sample in sublist]
    dif_tot = len(train_dat) - len(selected_samples)  # diff = how many samples do we have to generate
    diff = dif_tot // 10
    print("Creation of", dif_tot, "augmented samples...")
    print("Creation of", diff, "samples per class...")

    # create a list of transformation functions
    list_fun = [
      tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal_and_vertical")]),
      tf.keras.Sequential([tf.keras.layers.RandomRotation(0.2)]),
      tf.keras.Sequential([tf.keras.layers.RandomContrast(0.2)]),
      tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.2)])
    ]

    # apply transformation sequentially to generate additional samples
    transformed_samples = []
    selected_samples2 = selected_samples.copy()  # make a copy to avoid modifying the original list

    # calculate the number of additional samples to generate per iteration
    diff_per_iteration = int(len(selected_samples2) / 10)

    for i in range(10):
      partial = selected_samples2[:diff_per_iteration]  # get a slice of samples for this iteration
      del selected_samples2[:diff_per_iteration]  # remove the selected samples from the list

      samples_generated_this_iteration = 0  # track the number of generated samples in this iteration
      while samples_generated_this_iteration < diff:
        for img, label in partial:
          transformation_index = i % len(list_fun)  # cycle through the list of transformation functions
          transformed_img = list_fun[transformation_index](img)
          transformed_samples.append((transformed_img, label))

          samples_generated_this_iteration += 1  # increment the number of generated samples

          if samples_generated_this_iteration >= diff:
            break  # exit the inner loop if the desired number of samples is generated

    # combine original and new samples
    augmented_samples = selected_samples + transformed_samples
    print('AUGMENTATION COMPLETE')

    # convert the augmented samples into a TensorFlow dataset
    train_ds = tf.data.Dataset.from_tensor_slices((np.array([x[0] for x in augmented_samples], dtype=np.uint8), np.array([x[1] for x in augmented_samples], dtype=np.int64)))
    test_ds = tf.data.Dataset.from_tensor_slices((np.array([x[0] for x in test_dat], dtype=np.uint8), np.array([x[1] for x in test_dat], dtype=np.int64)))

  return train_ds, test_ds, info

# - Data Preprocessing Pipeline -------------------------------------------------
def prepare_data(data):  # normal preprocessing pipeline
  data = data.map(lambda image, target: (tf.cast(image, tf.float32), target))  # change the dtype from uint8 to float32
  data = data.map(lambda image, target: ((image / 128.0) - 1.0, target))  # normalize the values to a range from 1 to -1
  data = data.map(lambda image, target: (image, tf.one_hot(target, depth = 10)))  # one hot encode the class labels, we got 10 classes
  data = data.shuffle(1000).batch(128).prefetch(tf.data.experimental.AUTOTUNE)  # shuffle, batch and then prefetch
  return data

  

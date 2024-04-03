# IANNwTF_Final_Project - Improving ResNet Model Performance on EuroSAT-RGB Data through Data Augmentation



Choose one of the three Models, set up the variables “da” and “threshold”:

*threshold* represents how many training samples we are using from the original datasets (N.B. when = 2000 it comprehends also the test_set)

*da* represents how many original samples are we using to create augmented data (N.B. the training data size will be always generated based on the test_size parameter, if 0.3, training data size = 1400)

Base
da = "base" threshold = 2000

100 original samples per Class
da = "base" , threshold = 100

500 original samples per Class
da = "base" , threshold = 500

1000 original samples per Class
da = "base" , threshold = 1000

100 original sample + 1300 data aug
da = 100 threshold = 2000

500 original sample + 900 data aug
da = 500 threshold = 2000

1000 original sample + 400 data aug
da = 1000 threshold = 2000


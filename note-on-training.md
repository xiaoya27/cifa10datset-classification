note1: 


Q : 

[Higher validation accuracy, than training accurracy using Tensorflow and Keras](https://stackoverflow.com/questions/43979449/higher-validation-accuracy-than-training-accurracy-using-tensorflow-and-keras)

When training, a percentage of the features are set to zero (50% in your case since you are using Dropout(0.5)). 
When testing, all features are used (and are scaled appropriately). 
So the model at test time is more robust - and can lead to higher testing accuracies.

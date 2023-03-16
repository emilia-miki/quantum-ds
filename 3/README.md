# Task 3

This is my solution to the 3rd task. It contains the Jupyter Notebook 'main.ipynb'. In it you can find my analysis of the dataset, creation, training and exporting of a suitable model for the task at hand.

The file 'model.h5' contains the model itself, and 'targets_specs.txt' contains the mean and standard deviation of the targets in 'internship_train.csv', so that they can be used to reverse normalization of the targets after predicting them using the model.

These two files are used by the program main.py. It can take a command-line argument with the name of the csv file with data, or if the argument isn't provided it will ask for the filename in a cli prompt. After doing all the necessary preprocessing, using the model to get predictions an reversing their normalization, it appends a 'target' column to the provided file with the resulting predictions.

This program was used to provide predictions for the data in 'internship_hidden_test.csv', you can find them in the file itself it the 'target' column.

'requirements.txt' was generated using pipreqs.

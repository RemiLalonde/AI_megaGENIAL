from data_processing import DataProcessor
from model import SpikingNN
from training import Trainer
import torch

# Instantiate data processor
data_processor = DataProcessor()

# Load and process data
file_path_1 = 'Data/right_side_seizure_encode.csv'
file_path_2 = 'Data/right_side_no_seizure_encode.csv'
file_path_3 = 'Data/left_side_seizure_encode.csv'
file_path_4 = 'Data/left_side_no_seizure_encode.csv'
matrices = []
label_matrice = []

matrices, label_matrice = data_processor.load_csv_to_matrices(matrices, label_matrice, file_path_1, 0)
matrices, label_matrice = data_processor.load_csv_to_matrices(matrices, label_matrice, file_path_2, 1)
matrices, label_matrice = data_processor.load_csv_to_matrices(matrices, label_matrice, file_path_3, 2)
matrices, label_matrice = data_processor.load_csv_to_matrices(matrices, label_matrice, file_path_4, 1)

X_train, X_val, X_test, y_train, y_val, y_test = data_processor.prepare_data_for_ai(matrices, label_matrice)

# Prepare data for SNN
X_train, y_train = data_processor.prepare_data_for_snn(X_train, y_train)
X_val, y_val = data_processor.prepare_data_for_snn(X_val, y_val)
X_test, y_test = data_processor.prepare_data_for_snn(X_test, y_test)

# Instantiate and train the SNN model
snn_model = SpikingNN()

trainer = Trainer(snn_model, lr=0.001, patience=3)
train_losses, val_losses, train_accuracies, val_accuracies = trainer.train(X_train, y_train, X_val, y_val, epochs=60)

# Calculate and print accuracy on the test set
accuracy = trainer.evaluate(X_test, y_test)

# Plot and save the learning curve
trainer.plot_and_save_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path='./output/learning_curve_small_model_3_classes.png')

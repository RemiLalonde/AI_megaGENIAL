import torch
import json
from model import SpikingNN
from data_processing import DataProcessor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Custom encoder to convert tensors to lists for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ModelTester:
    def __init__(self, model_path, data_path_list):
        self.model_path = model_path
        self.data_path_list = data_path_list
        self.data_processor = DataProcessor()

    def load_model(self):
        """Load the saved model from the specified path."""
        model = SpikingNN()
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'),  weights_only=True))
        model.eval()
        return model

    def prepare_test_data(self):
        """Load and prepare all data for evaluation (without splitting into train/validation/test)."""
        matrices, labels = [], []

        for file_path, label in self.data_path_list:
            matrices, labels = self.data_processor.load_csv_to_matrices(matrices, labels, file_path, label, reverse_flag=False)

        matrices = torch.tensor(np.array(matrices), dtype=torch.float32)
        labels = torch.tensor(np.array(labels), dtype=torch.long)


        shuffled_indices = torch.randperm(matrices.size(0))  # Generate shuffled indices
        matrices = matrices[shuffled_indices]  # Reorder the matrices
        labels = labels[shuffled_indices]  # Reorder the labels

        return matrices, labels

    def test_model(self, save_info):
        """Load the model, prepare data, and evaluate the model's accuracy."""
        # Load the model
        model = self.load_model()

        # Prepare the test data
        X_test, y_test = self.prepare_test_data()

        # Perform predictions
        with torch.no_grad():
            outputs = model(X_test)
            print(outputs)
            _, predicted_labels = torch.max(outputs, 1)
            print(predicted_labels)
        # Compute accuracy
        correct = (predicted_labels == y_test).sum().item()
        accuracy = correct / len(y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')

        self.plot_confusion_matrix(y_test.numpy(), predicted_labels.numpy())

        self.plot_incorrect_predictions(X_test, y_test, predicted_labels)

        #self.save_inputs_outputs(X_test, predicted_labels, y_test)

        if save_info:
            # Save model and optimizer info to JSON
            self.save_model_info(model)

        return accuracy


    def save_model_info(self, model):
        """Save model parameters and state_dict to a JSON file."""
        dt = 1e-3
        model_info = {
            'state_dict': {param_tensor: model.state_dict()[param_tensor].tolist() for param_tensor in model.state_dict()},
        }

        layer_mapping = {
            'fc1': 'lif1',
            'fc2': 'lif2',
            'fc3': 'lif3',
            'fc4': 'lif4'
        }

        # Loop through the layer mappings and add the `beta` and `threshold` values under the new keys
        for fc_name, lif_name in layer_mapping.items():
            # Construct the new key names as `fc1.beta` and `fc1.threshold`
            beta_key = f'{fc_name}.beta'
            threshold_key = f'{fc_name}.threshold'
            reset_key = f'{fc_name}.reset'
            leak_key = f'{fc_name}.leak'

            # Extract the corresponding `beta` and `threshold` values from the LIF cell layer
            beta_value = torch.exp(-dt * getattr(model, lif_name).p.tau_mem_inv).item()
            threshold_value = getattr(model, lif_name).p.v_th.item()
            reset_value = getattr(model, lif_name).p.v_reset.item()
            leak_value = getattr(model, lif_name).p.v_leak.item()

            # Add these values to the `state_dict` under the new keys
            model_info['state_dict'][beta_key] = beta_value
            model_info['state_dict'][threshold_key] = threshold_value
            model_info['state_dict'][reset_key] = reset_value
            model_info['state_dict'][leak_key] = leak_value

        # Save information to a JSON file using a custom encoder
        with open('output/model_info.json', 'w') as json_file:
            json.dump({'model': model_info}, json_file, cls=NumpyEncoder)

        print("Model information saved to 'model_info.json'")

    @staticmethod
    def plot_confusion_matrix(true_labels, predicted_labels):
        """Compute and display confusion matrix."""
        cm = confusion_matrix(true_labels, predicted_labels)
        print(f'Confusion Matrix:\n{cm}')

        # Optionally, normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=[0, 1, 2, 3])  # Adjust class labels based on your dataset

        # Plot the confusion matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    @staticmethod
    def plot_incorrect_predictions(X_test, y_test, predicted_labels):
        """Plot incorrect predictions with columns as separate curves."""
        incorrect_indices = (predicted_labels != y_test).nonzero(as_tuple=True)[0]

        if len(incorrect_indices) == 0:
            print("No incorrect predictions.")
            return

        # Select some incorrect predictions to plot
        num_to_plot = min(5, len(incorrect_indices))  # Plot up to 5 incorrect predictions
        incorrect_samples = X_test[incorrect_indices[:num_to_plot]]
        true_labels = y_test[incorrect_indices[:num_to_plot]]
        predicted_labels_incorrect = predicted_labels[incorrect_indices[:num_to_plot]]

        fig, axes = plt.subplots(num_to_plot, 1, figsize=(10, num_to_plot * 3))
        if num_to_plot == 1:
            axes = [axes]  # Ensure axes is always iterable

        for i, ax in enumerate(axes):
            matrix = incorrect_samples[i].numpy()  # Convert tensor to numpy array
            num_rows = matrix.shape[0]  # Number of rows in the matrix
            num_columns = matrix.shape[1]  # Number of columns in the matrix

            # Plot each column as a separate line on the graph
            for col_idx in range(num_columns):
                ax.plot(range(num_rows), matrix[:, col_idx], label=f"Column {col_idx}")

            ax.set_title(f"True: {true_labels[i].item()}, Pred: {predicted_labels_incorrect[i].item()}")
            ax.set_xlabel("Row Index")
            ax.set_ylabel("Values")
            ax.legend(loc='upper right')

        plt.suptitle("Incorrect Predictions - Column Values")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Path to the saved model
    model_path = './output/output.pth'  # Path where the trained model is saved

    # List of file paths and corresponding labels for testing
    data_path_list = [
        ('Data/right_side_seizure_encode.csv', 0),
        ('Data/right_side_no_seizure_encode.csv', 1),
        ('Data/left_side_seizure_encode.csv', 2),
        ('Data/left_side_no_seizure_encode.csv', 3)
    ]

    data_path_list_Test = [
        ('Data/right_side_seizure_encode_test_set.csv', 0),
        ('Data/right_side_no_seizure_encode_test_set.csv', 1),
        ('Data/left_side_seizure_encode_test_set.csv', 2),
        ('Data/left_side_no_seizure_encode_test_set.csv', 3)
    ]
    # Instantiate the ModelTester and run the test
    tester = ModelTester(model_path, data_path_list_Test)
    tester.test_model(save_info=False)

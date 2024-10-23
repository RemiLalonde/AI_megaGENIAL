import numpy as np
from sklearn.model_selection import train_test_split
import torch

class DataProcessor:
    @staticmethod
    def reverse_matrix(matrix):
        """Reverses the order of the rows in the matrix."""
        return np.flipud(matrix)  # Flip upside-down (reverse rows)

    def load_csv_to_matrices(self, matrices, label_matrice, file_path, label, reverse_flag=True):
        """Load matrices from CSV and add both original and reversed versions."""
        with open(file_path, 'r') as file:
            data = file.readlines()

        current_matrix = []

        for line in data:
            stripped_line = line.strip()

            if stripped_line:
                row = [int(float(x)) for x in stripped_line.split(',') if x]
                current_matrix.append(row)
            else:
                if current_matrix:
                    matrix_np = np.array(current_matrix)
                    matrices.append(matrix_np)
                    label_matrice.append(label)

                    if reverse_flag:
                        # Add reversed version of the matrix
                        reversed_matrix = self.reverse_matrix(matrix_np)
                        matrices.append(reversed_matrix)
                        label_matrice.append(label)

                    current_matrix = []

        if current_matrix:
            matrix_np = np.array(current_matrix)
            matrices.append(matrix_np)
            label_matrice.append(label)

            # Add reversed version of the last matrix
            if reverse_flag:
                reversed_matrix = self.reverse_matrix(matrix_np)
                matrices.append(reversed_matrix)
                label_matrice.append(label)

        return matrices, label_matrice

    @staticmethod
    def prepare_data_for_ai(matrices, labels, test_size=0.05, val_size=0.3):
        matrices = np.array(matrices)
        labels = np.array(labels)
        matrices = np.array([matrix / (np.max(matrix) + 1e-8) for matrix in matrices])

        X_trainval, X_test, y_trainval, y_test = train_test_split(matrices, labels, test_size=test_size, shuffle=True, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_size, shuffle=True, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def prepare_data_for_snn(X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return X, y

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from early_stopping import EarlyStopping

class Trainer:
    def __init__(self, model, patience=10, lr=0.001, batch_size=64):
        self.model = model
        self.patience = patience
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        self.early_stopping = EarlyStopping(patience=self.patience)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, save_path="./output/model_testing.pth"):
        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        best_val_loss = float('inf')
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            self.model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                correct_train += self.calculate_batch_accuracy(outputs, batch_y) * len(batch_y)
                total_train += len(batch_y)

            train_loss = running_loss / len(train_loader)
            train_acc = correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            val_loss, correct_val, total_val = 0.0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_outputs = self.model(val_X)
                    val_loss += self.loss_fn(val_outputs, val_y).item()
                    correct_val += self.calculate_batch_accuracy(val_outputs, val_y) * len(val_y)
                    total_val += len(val_y)

            val_loss /= len(val_loader)
            val_acc = correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Best model saved with validation loss: {best_val_loss:.4f}")

            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

            if self.early_stopping.check_early_stop(val_loss):
                print("Early stopping triggered.")
                break

        return train_losses, val_losses, train_accuracies, val_accuracies

    @staticmethod
    def calculate_batch_accuracy(outputs, labels):
        _, predicted_labels = torch.max(outputs, 1)
        correct = (predicted_labels == labels).sum().item()
        accuracy = correct / len(labels)
        return accuracy

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(X_test)
            _, predicted_labels = torch.max(test_output, 1)
            correct = (predicted_labels == y_test).sum().item()
            accuracy = correct / len(y_test)
            print(f'Accuracy: {accuracy * 100:.2f}%')
            return accuracy

    @staticmethod
    def plot_and_save_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path='learning_curve_output.png'):
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f'Learning curve saved as {save_path}')
        plt.show()

import matplotlib.pyplot as plt
from snntorch import spikeplot as splt
import os
import csv

class Visualization():
    def save_train_val_loss(self, train_loss, val_loss):
        os.makedirs(f'images/val_train_loss', exist_ok=True)
        plt.figure()
        plt.plot(train_loss, label='train loss')
        plt.plot(val_loss, label='validation loss')
        plt.title('Train and validation loss')
        plt.xlabel('epochs')
        plt.ylabel('MSE loss')
        plt.legend()
        plt.savefig(f'images/val_train_loss/val_train_loss.png')


    def saveImage(self, tensor, input, epoch, num_cols, step, starting_column, positive, full=False):
    #print(tensor)
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        for i in range(num_cols):
            if input:
                if full:
                    plt.plot(tensor.detach().numpy()[:, i], label=f'input {i + 1}')
                else:
                    plt.plot(tensor.detach().numpy()[:, starting_column+ i], label=f'input {i + 1}')
            else:
                plt.plot(tensor.detach().numpy()[:, starting_column + i], label=f'output {i + 1}')
        # Add labels and legend
        plt.xlabel('Temps')
        plt.ylabel('Impulsion')
        plt.legend()

        # Show the plot
        if input:
            if full:
                plt.title("Entrée")
                plt.savefig(f'images/epoch_{epoch}/{step}/last_input_complete.png')
            elif positive:
                plt.title("Entrée positif")
                plt.savefig(f'images/epoch_{epoch}/{step}/last_input_positive.png')
            else:
                plt.title("Entrée")
                plt.savefig(f'images/epoch_{epoch}/{step}/last_input.png')
        else:
            plt.title("Sortie")
            plt.savefig(f'images/epoch_{epoch}/{step}/last_output.png')
        plt.close()

    def save_encoding(self, tensor, encoding, epoch, step):
    #print(tensor)
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        fig, axes = plt.subplots(2, 1)
        axes[0].plot(tensor, label='Input')
        axes[1].plot(encoding, label='Encoding')

        # Ajouter des légendes et des titres si nécessaire
        axes[0].set_title('Input')

        axes[1].set_title('Input encoded')

        plt.savefig(f'images/epoch_{epoch}/{step}/encoding.png')
        plt.close()
    
    def save_output(self, output, epoch, step, input):
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(output.detach().numpy()[:, 0:4], label='Positive')
        axes[1].plot(output.detach().numpy()[:, 4:8], label='Negative')

        axes[0].set_title('Positif')
        axes[1].set_title('Négatif')

        plt.subplots_adjust(hspace=0.5)
        fig.text(0.5, 0.04, 'Temps', ha='center', va='center')
        fig.text(0.04, 0.5, 'Impulsion', va='center', rotation='vertical')
        if input:
            plt.suptitle('Entrée encodé')
            plt.savefig(f'images/epoch_{epoch}/{step}/input.png')
        else:
            plt.suptitle('Sortie encodé')
            plt.savefig(f'images/epoch_{epoch}/{step}/output.png')
        plt.close()

    def save_encoding_positive_negative(self, tensor, encoding, epoch, step):
    #print(tensor)
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        fig, axes = plt.subplots(3, 1)
        axes[0].plot(tensor, label='Complete encoding')
        axes[1].plot(encoding[:, 0:4], label='Positive')
        axes[2].plot(encoding[:, 4:8], label='negative')
        # Ajouter des légendes et des titres si nécessaire
        axes[0].set_title('Encodage complet')
        axes[1].set_title('Positif')
        axes[2].set_title('Négatif')
        fig.text(0.5, 0.04, 'Temps', ha='center', va='center')
        fig.text(0.04, 0.5, 'Impulsion', va='center', rotation='vertical')
        plt.subplots_adjust(hspace=0.7)
        plt.suptitle('Encodage séparer en positif et négatif')
        plt.savefig(f'images/epoch_{epoch}/{step}/encoding_pos_neg.png')
        plt.close()

    def save_total(self, data, encoding, positive, epoch, step):
    #print(tensor)
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        fig, axes = plt.subplots(4, 1)
        axes[0].plot(data, label='Donnée')
        axes[1].plot(encoding, label='Encodage complet')
        axes[2].plot(positive[:, 0:4], label='Positif')
        axes[3].plot(positive[:, 4:8], label='Négatif')
        # Ajouter des légendes et des titres si nécessaire
        axes[0].set_title('Donnée')
        axes[1].set_title('Encodage complet')
        axes[2].set_title('Positif')
        axes[3].set_title('Négatif')
        plt.subplots_adjust(hspace=0.9)
        plt.savefig(f'images/epoch_{epoch}/{step}/complete_process.png')
        plt.close()

    def saveImageCombined(self, input_tensor, output_tensor, epoch, num_cols, step):
        os.makedirs(f'images/epoch_{epoch}/{step}/', exist_ok=True)

        # Create subplots
        fig, axs = plt.subplots(2, sharex=True)

        for i in range(num_cols):
            axs[0].plot(input_tensor.detach().numpy()[:, i], label=f'input {i + 1}')
            axs[1].plot(output_tensor.detach().numpy()[:, i], label=f'output {i + 1}')

        # Add labels and legend
        axs[0].set_ylabel('Input Value')
        axs[1].set_ylabel('Output Value')
        axs[1].set_xlabel('Row Index')
        # axs[0].legend()
        # axs[1].legend()

        plt.savefig(f'images/epoch_{epoch}/{step}/input_output.png')
        plt.close()

    def saveImageTestSet(self, input_tensor, output_tensor, window_number, num_cols):
        os.makedirs(f'images_test_set/test_number_{window_number}/', exist_ok=True)

        # Create subplots
        fig, axs = plt.subplots(2, sharex=True)

        for i in range(num_cols):
            axs[0].plot(input_tensor.detach().numpy()[:, i], label=f'input {i + 1}')
            axs[1].plot(output_tensor.detach().numpy()[:, i], label=f'output {i + 1}')

        # Add labels and legend
        axs[0].set_ylabel('Input Value')
        axs[1].set_ylabel('Output Value')
        axs[1].set_xlabel('Row Index')
        # axs[0].legend()
        # axs[1].legend()

        plt.savefig(f'images_test_set/test_number_{window_number}/input_output.png')
        plt.close() 

    def saveImageCombinedOnSame(self, input_tensor, output_tensor, epoch, num_cols, step):
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        # Create subplots
        plt.figure()

        for i in range(num_cols):
            plt.plot(input_tensor.detach().numpy()[:, i], color='blue')
            plt.plot(output_tensor.detach().numpy()[:, i], color='red')

        # Add labels and legend
        plt.ylabel('Value')
        plt.xlabel('Row Index')
        plt.legend(['Channel_input', 'Channel_output'])

        plt.savefig(f'images/epoch_{epoch}/{step}/input_output_superpose.png')
        plt.close() 

    def save_subplots(self, input, output, epoch, step):
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        num_channels = input.shape[1]

        fig, axs = plt.subplots(num_channels, 1, figsize=(8, 4 * num_channels))

        for i in range(num_channels):
            axs[i].plot(input[:, i].detach().numpy(), label=f'Input {i + 1}')
            axs[i].plot(output[:, i].detach().numpy(), label=f'Output {i + 1}', color='orange')
            axs[i].set_xlabel('Row Index')
            axs[i].set_ylabel('Value')
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(f'images/epoch_{epoch}/{step}/input_vs_output.png')
        plt.close()

    def save_subplotsTest(self, input, output, window_number):
        os.makedirs(f'images_test_set/test_number_{window_number}', exist_ok=True)

        num_channels = input.shape[1]

        fig, axs = plt.subplots(num_channels, 1, figsize=(8, 4 * num_channels))

        for i in range(num_channels):
            axs[i].plot(input[:, i].detach().numpy(), label=f'Input {i + 1}')
            axs[i].plot(output[:, i].detach().numpy(), label=f'Output {i + 1}', color='orange')
            axs[i].set_xlabel('Row Index')
            axs[i].set_ylabel('Value')
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(f'images_test_set/test_number_{window_number}/input_output_channel.png')
        plt.close()

    def save_test(self, data, window_number, input):
        fig, axes = plt.subplots(2, 1)

        axes[0].plot(data.detach().numpy()[:, 0:4], label='Positive')
        axes[1].plot(data.detach().numpy()[:, 4:8], label='Negative')

        axes[0].set_title('Positive')
        axes[1].set_title('Negative')

        if input:
            plt.savefig(f'images/test_number_{window_number}/input.png')
        else:
            plt.savefig(f'images/test_number_{window_number}/output.png')
        plt.close()
    
    def save_input_output_matrix(self, data, epoch, input):
        path = f'images/epoch_{epoch}/'
        os.makedirs(path, exist_ok=True)
        if input:
            file = f'{path}input.csv'
        else:
            file = f'{path}output.csv'
        with open(file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            for tensor in data:
                tensor_int = tensor.int().tolist()
                csv_writer.writerow(tensor_int)
            csv_writer.writerow([])

    def save_KMeans(self, data, centroids, num_clusters, cluster_labels, epoch, step, input):
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        for cluster_index in range(num_clusters):
            cluster_points = data[cluster_labels.flatten() == cluster_index]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_index}')

        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='*', s=300, label='Centroids')

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        # Show the plot
        if input:
            plt.savefig(f'images/epoch_{epoch}/{step}/last_input_kmeans.png')
        else:
            plt.savefig(f'images/epoch_{epoch}/{step}/last_output_kmeans.png')
        plt.close()

    def save_KMeans_latent(self, data, num_clusters, cluster_labels, epoch, step, input):
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        plt.figure(figsize=(8, 6))
        for i in range(num_clusters):
            plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], label=f'Cluster {i}')
        plt.title('K-means Clustering on Latent Space Representation')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # Show the plot
        if input:
            plt.savefig(f'images/epoch_{epoch}/{step}/last_input_kmeans_latent.png')
        else:
            plt.savefig(f'images/epoch_{epoch}/{step}/last_output_kmeans_latent.png')
        plt.close()

    def save_raster(self, data, epoch, step, input):
        os.makedirs(f'images/epoch_{epoch}/{step}', exist_ok=True)

        fig = plt.figure(facecolor="w", figsize=(10,5))
        splt.raster(data[-1,:,:], ax=fig.add_subplot(111), s=1.5, c="black")
        plt.title("Raster Plot of the last window")
        plt.xlabel("Time step")
        plt.ylabel("Neuron Number")
        
        # Show the plot
        if input:
            plt.savefig(f'images/epoch_{epoch}/{step}/last_input_raster.png')
        else:
            plt.savefig(f'images/epoch_{epoch}/{step}/last_output_raster.png')
        plt.close()


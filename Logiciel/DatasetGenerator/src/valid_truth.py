import numpy as np
import spikeinterface.extractors as se
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, silhouette_score, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

recording, ground_truth = se.read_mearec('data/recordings/recordings_tetrode-mea-s_2min.h5')

unit_ids = ground_truth.get_unit_ids()

spike_times = []
spike_units = []

for unit_id in unit_ids:
    spikes = ground_truth.get_unit_spike_train(unit_id)
    if spikes.size > 0:
        spike_times.extend(spikes)
        spike_units.extend([int(unit_id.lstrip('#'))] * len(spikes))

data = np.column_stack((spike_times, spike_units))

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

n_clusters = len(set(spike_units))  # Number of unique ground truth units
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(data_normalized)

label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(spike_units)

ground_truth_labels = np.array(spike_units)

conf_matrix = confusion_matrix(ground_truth_labels, kmeans_labels)

row_ind, col_ind = linear_sum_assignment(-conf_matrix)  # maximizing the accuracy

accuracy = conf_matrix[row_ind, col_ind].sum() / conf_matrix.sum()

print(f"Clustering Accuracy: {accuracy:.4f}")

kmeans_labels_aligned = kmeans_labels.copy()
for k, new_k in enumerate(col_ind):
    kmeans_labels_aligned[kmeans_labels == new_k] = row_ind[k]

precision = precision_score(true_labels_encoded, kmeans_labels_aligned, average='weighted')

print(f"Precision: {precision:.4f}")

plt.figure(figsize=(10, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=np.unique(ground_truth_labels))
disp.plot(cmap='Greens')
plt.title('Ground Truth Validation')
plt.ylabel('Ground Truth Labels')
plt.xlabel('K-Means Labels')

label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(spike_units)
silhouette = silhouette_score(data_normalized, kmeans_labels)
print(f"Silhouette Score: {silhouette:.4f}")

plt.show()
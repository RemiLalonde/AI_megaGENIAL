import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import spikeinterface as si
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

recording, ground_truth = se.read_mearec('data/recordings/recordings_tetrode-mea-s_2min.h5')

unit_ids = ground_truth.get_unit_ids()

spike_times = []
spike_units = []

for unit_id in unit_ids:
    spikes = ground_truth.get_unit_spike_train(unit_id)  # Get spike times for the unit
    if spikes.size > 0:  # Check if there are spikes for the unit
        spike_times.extend(spikes)  # Add spike times to the list
        spike_units.extend([int(unit_id.lstrip('#'))] * len(spikes))  # Convert to int

if len(spike_times) == len(spike_units):
    data = np.column_stack((spike_times, spike_units))
else:
    raise ValueError("Mismatch between number of spikes and unit IDs")

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

n_clusters = len(set(spike_units))  # Number of unique ground truth units
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data_normalized)

labels = kmeans.labels_

print("Cluster labels:", labels)

silhouette_avg = silhouette_score(data_normalized, labels)
print(f'Silhouette Score: {silhouette_avg:.3f}')

for i in range(n_clusters):
    print(f"Cluster {i}: {np.sum(labels == i)} spikes")

plt.figure(figsize=(10, 6))

for cluster in range(n_clusters):
    plt.scatter(data[:, 0][labels == cluster], data[:, 1][labels == cluster],
                label=f'Regroupement {cluster}', s=25, alpha=0.7)

plt.title('Validation par regroupement K-Means')
plt.xlabel('Échantillons')
plt.ylabel('Canaux encodés positifs et négatifs')

plt.xscale('log')

plt.legend()
plt.tight_layout()
plt.show()

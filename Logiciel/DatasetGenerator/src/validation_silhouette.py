import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


recording, ground_truth_sorting = se.read_mearec('data/recordings/recordings_tetrode-mea-s_2min.h5')

waveform_extractor = st.extract_waveforms(recording, ground_truth_sorting, 'output_folder')

pca = PCA(n_components=2)
features = pca.fit_transform(waveform_extractor.get_waveforms())

labels = ground_truth_sorting.get_labels()

score = silhouette_score(features, labels)
print(f'Silhouette Score: {score}')
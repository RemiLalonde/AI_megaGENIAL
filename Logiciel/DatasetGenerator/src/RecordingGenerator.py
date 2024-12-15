import MEArec as mr
from pprint import pprint
import matplotlib.pylab as plt

class RecordingGenerator:
    def __init__(self, minDist=10, nbrOfInh=5, nbrOfExc=5, 
                 overlap=True, duration=30, filter=False):
        self.recordings_params = mr.get_default_recordings_params()
        self.minDist = minDist
        self.nbrOfInh = nbrOfInh
        self.nbrOfExc = nbrOfExc
        self.overlap = overlap
        self.duration = duration
        self.filter = filter
        self.recordings_params['min_dist'] = minDist
        self.recordings_params['spiketrains']['n_exc'] = nbrOfExc
        self.recordings_params['spiketrains']['n_inh'] = nbrOfInh
        self.recordings_params['recordings']['overlap'] = overlap
        self.recordings_params['recordings']['filter'] = filter
        self.recordings_params['spiketrains']['duration'] = duration

        #Seed pour reproduire les meme signaux
        self.recordings_params['seeds']['convolution'] = 30
        self.recordings_params['seeds']['noise'] = 30
        self.recordings_params['seeds']['spiketrains'] = 30
        self.recordings_params['seeds']['templates'] = 30
         
    def setMinDist(self, minDist):
        self.minDist = minDist
        self.recordings_params['min_dist'] = minDist

    def setNbrOfInh(self, nbrOfInh):
        self.nbrOfEAP = nbrOfInh
        self.recordings_params['spiketrains']['n_inh'] = nbrOfInh

    def setNbrOfExc(self, nbrOfExc):
        self.nbrOfEAP = nbrOfExc
        self.recordings_params['spiketrains']['n_exc'] = nbrOfExc
    
    def setNbrOfExc(self, overlap):
        self.overlap = overlap
        self.recordings_params['recordings']['overlap'] = overlap

    def setNbrOfExc(self, duration):
        self.duration = duration
        self.recordings_params['spiketrains']['duration'] = duration
    
    def setNbrOfExc(self, filter):
        self.filter = filter
        self.recordings_params['recordings']['filter'] = filter

    def generateRecording(self, noise=False, probeName=""):
        pprint(self.recordings_params)

        #Permet de generer different niveau de bruit
        if noise:
            noise_levels = [5, 100]
            recgen_list = []

            for n in noise_levels:
                print('Noise level: ', n)
                self.recordings_params['recordings']['noise_level'] = n
                self.recordings_params['spiketrains']['duration'] = 0.5
                self.recordings_params['recordings']['filter'] = False
                recgen = mr.gen_recordings(templates='Logiciel/DatasetGenerator/src/data/template/templates.h5', 
                                           params=self.recordings_params, 
                                           verbose=False)
                recgen_list.append(recgen)

                for recgen in recgen_list:
                    mr.plot_recordings(recgen, start_time=0, end_time=1, 
                                       overlay_templates=True, cmap='cool')
                    mr.plot_recordings(recgen, cmap='cool') 
        else:
            recgen = mr.gen_recordings(templates='Logiciel/DatasetGenerator/src/data/template/templates.h5', 
                                       params=self.recordings_params, 
                                       verbose=False)

            mr.plot_recordings(recgen, start_time=0, end_time=1, 
                               overlay_templates=True, cmap='cool')
            mr.plot_waveforms(recgen, electrode='max', cmap='cool')
            mr.plot_recordings(recgen, cmap='cool') 

        mr.save_recording_generator(recgen, filename=f'Logiciel/DatasetGenerator/src/data/recordings/recordings_{probeName}_10s.h5')

        mr.plot_templates(recgen, single_axes=False, ncols=4, cmap='cool')
        mr.plot_templates(recgen, single_axes=True, cmap='cool')
        mr.plot_rasters(recgen.spiketrains)
        mr.plot_pca_map(recgen, cmap='cool')

        # for i in range(0,400):
        #     print(recgen.recordings[i])
        print('Recordings shape', recgen.recordings.shape)
        print('Sample spike train', recgen.spiketrains[0])
        print('Sample spike train', recgen.spiketrains[0].size)
        print('Sample spike train', recgen.spiketrains[1].size)
        print('Sample spike train', recgen.spiketrains[2].size)
        print('Sample spike train', recgen.spiketrains[3].size)
        plt.show(block=True)

def main():
    rg = RecordingGenerator(nbrOfInh=2, nbrOfExc=2, overlap=False, 
                            duration=10, filter=False)
    rg.generateRecording(probeName="tetrode-mea-s")

if __name__ == "__main__":
    main()


import MEAutility as MEA
import matplotlib.pylab as plt
import numpy as np

tetrode = MEA.return_mea('tetrode-mea-s')

MEA.plot_probe(tetrode)

plt.show(block=True)

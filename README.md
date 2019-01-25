# MultiOutputSpectralMixture
This repository hosts the code for

G. Parra, F. Tobar, Spectral Mixture Kernels for Multi-Output Gaussian Processes, Advances in Neural Information Processing Systems, 2017

Proceedings link: http://papers.nips.cc/paper/7245-spectral-mixture-kernels-for-multi-output-gaussian-processes

#Requirements
-Python 3.6
-GPFlow 1.3

#Notes on requirements
It seems that the latest tensorflow packages can be found in the anaconda channel, so using the
following command to create a new conda environments yields the easiest results:
	conda create -n TFGPU -c anaconda scipy scikit-learn matplotlib seaborn tensorflow-gpu 
This command creates a new env named TFGPU with tensorflow-gpu 1.12 (as of this writing).
--------------------------------------------------------------------------------------------------
To install GPFlow: 
- Clone the repo https://github.com/GPflow/GPflow
- source the environment TFGPU
- Go to GPflow folder
- run pip install .

The above commands should work as long as the pip you're using corresponds to the env you 
created. To make sure this is the case you can use the command 'which pip', and this should
should a pip running inside a bin folder localed at the environment you created. Otherwise, you'd
be using other pip (which could be from another env or, even worse, from the local machine).

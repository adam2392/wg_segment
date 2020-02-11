# Installation Instructions

Setup basic environment via Conda:

    conda create -n wg_segment
    conda activate wg_segment
    conda config --add channels conda-forge
    conda install numpy pandas mne scipy scikit-learn seaborn matplotlib
    conda install mne mne-bids 
    conda install natsort xlrd deprecated tqdm pybv
    conda install pytorch torchvision -c pytorch
    
    # tslearn
    pip install tslearn
    
    # dmd
    pip install pydmd
    
    # dev versions of mne-python, mne-bids for easier loading and handling of data
    pip install --upgrade --no-deps https://api.github.com/repos/mne-tools/mne-python/zipball/master
    pip install --upgrade https://api.github.com/repos/mne-tools/mne-bids/zipball/master
    
To install Randomer Forests, follow instructions at: https://rerf.neurodata.io/install.html


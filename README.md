# NEURO120 final project

## "Use of Occlusion Methods on 4D Convolutional Neural Networks to Find Neural Representations of Perceived Emotions in fMRI Recordings"

This is my final projedt for the course NEURO120: Computational Neuroscience at Harvard College in the Spring 2025 semester. The project report exists in the repository as *projectpaper.pdf*. The project is made possible thanks to the StudyForrest dataset available publicly on https://www.studyforrest.org/.

### Repo structure
The repo consists of the following files:
**config.py** -- python file containing all imported libraries as well as the 3D CNN model used throughout the project
**dataprep.ipynb** -- jupyter notebook containing the original data preprocesing before neural network training
**training.ipynb** - jupyter notebook containting the training and testing loop as well as AOC analysis -- *notebook adjusted for Google Colab*
**occlusion.ipynb** - jupyter notebook running occlusion analysis on the validation dataset and computing UMAP clustering -- *notebook adjusted for Google Colab*
**rsaanalysis.ipynb** - jupyter notebook running RDM analysis of the CNN and fMRI data
**valindices.pkl** - pickle file containing the indices of validation dataset within the broader metadata file
**Xmetadata.pkl** - metadata of all fMRI volumes ready for training
**meta_val_filtered.pkl** - metadata filtered for occlusion training (only volumes from the validation dataset with some threshold accuracy)
**bestmodel.pth** - torch file containing the saved weights from best model training iteration
**readydata/** - folder containing the various .pt files developed in dataprep.ipynb before moving to training.ipynb

### Necessary to run
The model runs off data from the StudyForrest open source dataset. Below is a directory tree of the used files from the dataset (obtained by a combination of Datalad get & direct download from OpenNeuro)

.
└── Data/
    ├── ds000113/
    │   └── sub-**/
    │       └── ses-movie/
    │           └── func/
    │               ├── sub-**_ses-movie_task-movie_run-1_bold.nii.gz
    │               ├── sub-**_ses-movie_task-movie_run-2_bold.nii.gz
    │               ├── sub-**_ses-movie_task-movie_run-3_bold.nii.gz
    │               ├── sub-**_ses-movie_task-movie_run-4_bold.nii.gz
    │               ├── sub-**_ses-movie_task-movie_run-5_bold.nii.gz
    │               ├── sub-**_ses-movie_task-movie_run-6_bold.nii.gz
    │               ├── sub-**_ses-movie_task-movie_run-7_bold.nii.gz
    │               └── sub-**_ses-movie_task-movie_run-8_bold.nii.gz
    └── gump_emotions/
        └── raw/
            ├── av1o01
            ├── av1o02
            ├── av1o03
            ├── av1o04
            ├── av1o05
            ├── av1o06
            ├── av1o07
            ├── av1o08
            └── av1o09


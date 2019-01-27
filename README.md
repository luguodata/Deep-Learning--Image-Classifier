# Deep-Learning--Image-Classifier
This a project about an image classifier application via deep learning with PyTorch. It is from Udacity Data Science Nano-Degree program.

## Getting Started
Project mainly about two parts.
1) Training implementation on Jupyter notebook.
2) Python application which runs from the command line.

Files:
Image_Classifier_Application.ipynb -- Jupyter notebook of Training and testing image classifier.

Image_Classifier_Application.html  -- Html file of Training and testing image classifier.

train.py -- Python application which is used for training the image classifier network.
            Usages: * python train.py data_directory
                    * It will print out training loss, validation loss, and validation accuracy as the network trains
                    * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
                    * Choose architecture: python train.py data_dir --arch "vgg16"
                    * Set hyperparameters: python train.py data_dir --learning_rate 0.005 --hidden_units 512 --epochs 12
                    * Use GPU for training: python train.py data_dir --gpu

predict.py -- Python application which is used for predict flower name from an image along with the probability.
              Usage: * python predict.py /path/to/image checkpoint (checkpoint is the one saved from Image_Classifier_Application.ipynb)
                     * Return top $K$ most likely classes: python predict.py input checkpoint --top_k 3
                     * Use a mapping of categories to real names: python predict.py input checkpoint --category_names     
                       cat_to_name.json
                     * Use GPU for inference: python predict.py input checkpoint --gpu

Functions.py -- Functions and classes relating to data loading, image processing and model building.

### Prerequisites

### Installing

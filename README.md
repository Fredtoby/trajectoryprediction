![](https://img.shields.io/github/issues/rohanchandra30/Spectral-Trajectory-Prediction?color=brightgreen&style=plastic)![](https://img.shields.io/github/stars/rohanchandra30/Spectral-Trajectory-Prediction?color=cyan&style=plastic)![](https://img.shields.io/github/forks/rohanchandra30/Spectral-Trajectory-Prediction?color=blue&style=plastic)

### Paper - [**Forecasting Trajectory and Behavior of Road-Agents Using Spectral Clustering in Graph-LSTMs**](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9126166)

Project Page - https://gamma.umd.edu/spectralcows

Please cite our work if you found it useful.

```
@article{chandra2020forecasting,
  title={Forecasting trajectory and behavior of road-agents using spectral clustering in graph-lstms},
  author={Chandra, Rohan and Guan, Tianrui and Panuganti, Srujan and Mittal, Trisha and Bhattacharya, Uttaran and Bera, Aniket and Manocha, Dinesh},
  journal={IEEE Robotics and Automation Letters},
  year={2020},
  publisher={IEEE}
}
```

<p align="center">
<img src="figures/predict.png" width="260">
<img src="figures/results.gif" width="266">
<img src="figures/behavior.gif" width="260">
</p>

**Important** - This repo is no longer under active maintenance. Also, please note that the current results produced by the code are normalized RMSE values and not in meters. Furthermore, the trained models provided by in this codebase may not reflect the results in the main paper.


Table of Contents
=================

  * [Paper - <a href="https://obj.umiacs.umd.edu/gamma-umd-website-imgs/pdfs/autonomousdriving/spectralcows_full.pdf" rel="nofollow"><strong>Forecasting Trajectory and Behavior of Road-Agents Using Spectral Clustering in Graph-LSTMs</strong></a>](#paper---forecasting-trajectory-and-behavior-of-road-agents-using-spectral-clustering-in-graph-lstms)
  * [**Repo Details and Contents**](#repo-details-and-contents)
     * [List of Trajectory Prediction Methods Implemented](#list-of-trajectory-prediction-methods-implemented)
     * [Datasets](#datasets)
  * [**How to Run**](#how-to-run)
     * [Installation](#installation)
     * [Usage](#usage)
     * [Data Preparation.](#data-preparation)
        * [Formatting the dataset after downloading from the official website](#formatting-the-dataset-after-downloading-from-the-official-website)
        * [For preparing the formatted data into the data structures which our model requires](#for-preparing-the-formatted-data-into-the-data-structures-which-our-model-requires)
     * [Training and Testing on your own dataset](#training-and-testing-on-your-own-dataset)
        * [1. Prepare your Dataset](#1-prepare-your-dataset)
        * [2. Convert the text file to .npy format and save this as TrainSet0.npy.](#2-convert-the-text-file-to-npy-format-and-save-this-as-trainset0npy)
        * [3. Run the data_stream.py file in /data_processing. This will generate the pickle files needed to run the main.py files for any method.](#3-run-the-data_streampy-file-in-data_processing-this-will-generate-the-pickle-files-needed-to-run-the-mainpy-files-for-any-method)
        * [4. Then run the main.py file of any method.](#4-then-run-the-mainpy-file-of-any-method)
  * [**Our network**](#our-network)

## Repo Details and Contents
Python version: 3.7

### List of Trajectory Prediction Methods Implemented
Please cite the methods below if you use them.

* [**TraPHic: Trajectory Prediction in Dense and Heterogeneous Traffic Using Weighted Interactions**, CVPR'19](https://gamma.umd.edu/researchdirections/autonomousdriving/traphic/)<br>
Rohan Chandra, Uttaran Bhattacharya, Aniket Bera, Dinesh Manocha.
* [**Convolutional Social Pooling for Vehicle Trajectory Prediction**, CVPRW'18](https://arxiv.org/pdf/1805.06771.pdf)<br>
Nachiket Deo and Mohan M. Trivedi.
* [**Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks**, CVPR'18](https://arxiv.org/pdf/1803.10892.pdf)<br>
Agrim Gupta, Justin Johnson, Fei-Fei Li, Silvio Savarese, Alexandre Alahi.
* [**GRIP: Graph-based Interaction-aware Trajectory Prediction**, ITSC'19](https://arxiv.org/pdf/1907.07792.pdf)<br>
Xin Li, Xiaowen Ying, Mooi Choo Chuah 

As the official implementation of the GRIP method was not available at the time creating this repo, the code provided here is our own effort to replicate the GRIP method to the best of our ability and does not necessarily convey the original implementation of the authors. 

The original GRIP implementation by the authors is provided [here](https://github.com/xincoder/GRIP). Please cite their paper if you use their method. 


### Datasets
* [**Argoverse**](https://www.argoverse.org/data.html) (input length: 20 & output length: 30)
* [**Apolloscape**](http://apolloscape.auto/trajectory.html) (input length: 6 & output length: 10)
* [**Lyft Level 5**](https://level5.lyft.com/dataset/) (input length: 20 & output length: 30)


## How to Run

### Installation
---
1. Create a conda environement<br>
  `conda env create -f env.yml`

2. To activate the environment:<br>
  `conda activate sc-glstm`

3. Download resources <br>
  `python setup.py`

### Usage
---
* To run our one & two stream model:<br>
  1. `cd ours/`<br>
  2. `python main.py`
  3. To change between one stream to two stream, simply change the variable `s1` in main.py between True and False.
  4. To change the model, change `DATA` and `SUFIX` variable in main.py.
* To run EncDec comparison methods:<br>
  1. `cd comparison_methods/EncDec/`<br>
  2. `python main.py`
  3. To change the model, change `DATA` and `SUFIX` variable in main.py.
* To run GRIP comparison methods:<br>
  1. `cd comparison_methods/GRIP/`<br>
  2. `python main.py`
  3. To change the model, change `DATA` and `SUFIX` variable in main.py.
* To run TraPHic/SC-LSTM comparison methods:<br>
  1. `cd comparison_methods/traphic_sconv/`
  2. `python main.py`
  3. To change the model and methods, change `DATASET` and `PREDALGO` variable in main.py.

Note: During evaluation of the trained_models, the best results may be different from reported error due to different batch normalization applied to the network. To obtain the same number, we may have to mannually change the network.

Resources folder structure:
* data -- input and output of stream 1 & 2 (This is directly avaiable in resources folder)
* raw_data -- location of the raw data (put the downloaded dataset in this folder to process)
* trained_model -- some saved models


### Data Preparation.
---
Important steps if you plan to prepare the Argoverse, Lyft, and Apolloscape from the raw data available from their websites.

#### Formatting the dataset after downloading from the official website
* Run `data_processing/format_apolloscape.py` to format the downloaded apolloscape data into our desired representation
* Run `data_processing/format_lyft.py` to format the downloaded lyft data into our desired representation
* Run `data_processing/generate_data.py` to format the downloaded Argoverse trajectory data into our desired representation

#### For preparing the formatted data into the data structures which our model requires
* Use `data_processing/data_stream.py` to generate input data for stream1 and stream2. 
* Use `generate_adjacency()` function in `data_processing/behaviors.py` to generate adjacency matrices.
* Must use `add_behaviors_stream2()` function in `data_processing/behaviors.py` to add behavior labels to the stream2 data before supplying the data to the network.

### Training and Testing on your own dataset
---
Our code supports any dataset that contains trajectory information. Follow the steps below to integrate your dataset with our code

#### 1. Prepare your Dataset
The first step is to prepare your dataset in our format which is a text file where each row will contain 'Frame ID', 'Agent_ID', 'X coordinate', 'Y Coordinate', 'Dataset_ID'.

Make sure:
- The Frame_ID's range between `1 to n`. And Agent_ID's also range from `1 to N`. `n` is total number of frames and `N` is total number of agents. If your dataset uses a different convention to represent the Frame_ID's (for example, few datasets use Time Stamp as Frame_ID), you need to map these ID's to `1 to n`. If your dataset uses a different convention to represent Agent_ID's (for example few datasets represent Agent_ID's using string of characters), you need to map these ID's to `1 to N`. 

- If the Frame_ID's and Agent_ID's of your dataset are already in ranges of `1 to n` and `1 to N`, make sure they are sequential. Make sure there are no missing ID's. 

<!--- The text file is formed in such a way that Frame_ID's are in increasing order, starting from 1 to n. To double check, if you have successfully formatted your data into our format, the first few rows will have repeated Frame_ID's. The corresponding Agent_ID's of these repeated Frame_ID's must not be repeating.-->   

- Dataset_ID's are used to differentiate different scenes/sets of a same DATASET

#### 2. Convert the text file to `.npy` format and save this as `TrainSet0.npy`.

#### 3. Run the `data_stream.py` file in /data_processing. This will generate the pickle files needed to run the `main.py` files for any method.

Mandatory precautions to take before running `data_stream.py`:

- Make sure you have taken all the mandatory precautions mentioned above for preparing your data.

- You must know the frame rate at which the trajectories of the vehicles are recorded. i.e., you must know how many frames does 1 second corresponds to? E.g. if the FPS is 2Hz, this means each second corresponds to 2 frames in the dataset.

- You must set the `train_seq_len` and `pred_seq_len` in `data_stream.py` appropriately based on the frame rate. For example, if the frame rate is 2Hz, and if you want to consider 3 seconds as observation data, then `train_seq_len` would be `3*2 = 6`. if you want the to consider next 5 seconds as prediction data, then `pred_seq_len` would be `5*2 = 10`. Make sure `frame_lenth_cap >= (train_seq_len + pred_seq_len)`. We use this `frame_lenth_cap` to enforce that an `Agent_ID` is present/visible/seen in atleast `frame_lenth_cap` number of frames. 

- If your data is too huge, you may want to consider only few scenes/sets from the whole data. Use the Dataset IDs (`D_id`) list to tweak the values and shorten the amount of data.

- Assign a short keyword `XXXX` for naming your dataset.

- Expect to see multiple files generated in the `./resources/DATA/XXXX/ ` with names starting with `stream1_obs_data_, stream1_pred_data_, stream2_obs_data_, stream2_pred_data_,stream2_obs_eigs_, stream2_pred_eigs_`.

#### 4. Then run the `main.py` file of any method.

<!--### Plotting

* use the `plot_behaviors()` function in `data_processing/behaviors.py` to plot the behaviors of the agents.-->

## Our network

<p align="center">
<img src="figures/network.png">
</p>

<!--
### Trajectory Prediction Result
![Trajectory Prediction Result](figures/spectral_cluster_regularization.png)

## Comparison with other models
![comparison of our methods with other methods](figures/compare.png)

### Behavior prediction results
<p align="center">
  <img src="figures/behaviors.png">
</p>
-->


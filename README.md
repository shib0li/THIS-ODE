# *THIS-ODE*: Decomposing Temporal High-Order Interactions via Latent ODEs

by [Shibo Li](https://imshibo.com), [Mike Kirby](https://www.cs.utah.edu/~kirby/) and [Shandian Zhe](https://www.cs.utah.edu/~zhe/)

<p align="center">
    <br>
    <img src="images/THIS-ODE.png" width="500" />
    <br>
<p>

<h4 align="center">
    <p>
        <a href="https://proceedings.mlr.press/v162/li22i.html">Paper</a> |
        <a href="https://github.com/shib0li/THIS-ODE/blob/main/images/slides-v2.pdf">Slides</a> |
        <a href="https://github.com/shib0li/THIS-ODE/blob/main/images/923-poster.png">Poster</a> 
    <p>
</h4>

We propose a novel Temporal High-order Interaction decompoSition model based on Ordinary Differential Equations (**THIS-ODE**). We model the time-varying interaction result with a latent ODE. To capture the complex temporal dynamics, we use a neural network (NN) to learn the time derivative of the ODE state. We use the representation of the interaction objects to model the initial value of the ODE and to constitute a part of the NN input to compute the state. In this way, the temporal relationships of the participant objects can be estimated and encoded into their representations. For tractable and scalable inference, we use forward sensitivity analysis to efficiently compute the gradient of ODE state, based on which we use integral transform to develop a stochastic mini-batch learning algorithm.

# System Requirements

We highly recommend to use Docker to run our code. We have attached the docker build file `env.Dockerfile`. Or feel free to install the packages with pip/conda that could be found in the docker file.

# Datasets

In our paper, we conduct our experiments on 4 real-world tasks and 1 synthetic task. For each task, we evaluate all the methods on two settings: ***extrapolation*** and ***interpolation***. 

You should download original raw datasets([FitRec](https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local), [Time series events](https://github.com/snudatalab/TATD)) into the `data/raw/(domain)` folders and use the notebooks provided to process the data. If propoerly processed, the data used for training and testing should appear as the `data/processed/BeijingAirExtrap.pickle` for example

```
data/
├── raw/
│   └── Beijing/
│       └──...
│   └── FitRec/
│       └──...
├── processed
├── (notebooks)
```

# Run

To run all the methods, use our provided `run.sh` script, this script takes several arguments

```
./run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEVICE $FOLD $BATCH_SIZE $TEST_INTERVAL 

```

* `$DOMAIN` is a concat string that specifies the task and setting, for example ***BeijingAirExtrap*** means run with BeijingAir data on extrapolation setting.
* `$METHOD` learning algorithms
* `$MAX_EPOCHS` an integer of number of epochs
* `$RANK` dimension used for decomposition representation
* `$DEVICE` where to run, for example ***cuda:0*** or ***cpu***
* `$FOLD` fold index after splitting the datasets
* `$BATCH_SIZE` mini-batch size used for training
* `$TEST_INTERVAL` frequency for saving the results

for example

```
bash run.sh BeijingAirExtrap Neural_time 10 8 cuda:0 0 100 1 
```

runs the experiment of BeijingAir on the extraplation setting with the algorihtm of neural_time for 10 epochs. The dimensional of representation is 8 and the program runs on the first GPU. The minibatch size is 100 and save the reulst every 1 update.

Here are the name mappings of the names we used in the paper and the methods we used in the code.

| Names in the paper | Names in this repo |
|:------------------:|:------------------:|
|       CP-Time      |      CPTF_time     |
|       CP-DTL       |     CPTF_linear    |
|       CP-DTN       |      CPTF_rnn      |
|      GPTF-Time     |      GPTF_time     |
|      GPTF-DTL      |     GPTF_linear    |
|      GPTF-DTN      |      GPTF_rnn      |
|      NTF-Time      |     Neural_time    |
|       NTF-DTL      |    Neural_linear   |
|       NTF-DTN      |     Neural_rnn     |
|       PTucker      |       Tucker       |
|      THIS-ODE      |     NODE_noise     |

# License

THIS-ODE is released under the MIT License, please refer the LICENSE for details

# Getting Involved
Feel free to submit Github issues or pull requests. Welcome to contribute to our project!

To contact us, never hestitate to send an email to `shibo@cs.utah.edu` or `shiboli.cs@gmail.com` 
<br></br>


# Citation
Please cite our paper if you find it helpful :)

```

@InProceedings{pmlr-v162-li22i,
  title = 	 {Decomposing Temporal High-Order Interactions via Latent {ODE}s},
  author =       {Li, Shibo and Kirby, Robert and Zhe, Shandian},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {12797--12812},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/li22i/li22i.pdf},
  url = 	 {https://proceedings.mlr.press/v162/li22i.html},
}

```
<br></br>

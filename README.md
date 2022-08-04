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

We highly recommend to use Docker to run our code. We have attached the docker build file `env.Dockerfile`. Or feel free to install the packages with pip/conda that listed in the docker file.


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
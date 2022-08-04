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


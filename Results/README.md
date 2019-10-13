## Results

In this study, we had two research question:

1- Which of the two metrics (Surprise Coverage or Label Change Rate) is more effective in detecting adversarial examples?

2- How sensitive the two metrics are with respect to changing parameters?


### RQ1 Conclusion:
Our study shows that the conclusion for RQ1 is when applying LCR and LSC, using the default or suggested parameters, on two DNN models on MNIST, Label Change Rate (LCR), which is based on model mutation, is more effective than Likelihood Surprise Coverage (LSC), which is based on neuron coverage, in detecting adversarial examples.


#### Figures 1, 2 are comparing two metrics for MNIST dataset, LENET and 5-layer convolutional network and 5 attacks.
<img src="/Results/MNIST.png" width="800" height="800">


#### Figures 3, 4 are comparing two metrics for CIFAR-10 dataset, GOOGLENET and 12-layer convolutional network and 5 attacks.
<img src="/Results/CIFAR.png" width="800" height="800">


### RQ2 Conclusion:
Also, to answer RQ2, we have analysed different parameters: 
-Change Mutation Rate
-Change Layer
-Change Mutation operator

As shown in figures 5,6 and 7, we can confirm that that both metric are quite sensitive to their parameters and none is better than other in all cases. Even the individual performance can be really affected to the point that with a wrong parameter a metric would not be predictive anymore. Therefore, hyper-parameter tuning is very important for testing tools and metrics in this domain.
 
 #### Change Layer Effect
 <img src="/Results/ChangeLayer.png" width="700" height="500">
 
 #### Change Mutation Rate Effect
 <img src="/Results/ChangeMutationRate.png" width="700" height="500">
 
 #### Change Mutation Operator(NS and WS) Effect
 <img src="/Results/ChangeOperator.png" width="700" height="1000">

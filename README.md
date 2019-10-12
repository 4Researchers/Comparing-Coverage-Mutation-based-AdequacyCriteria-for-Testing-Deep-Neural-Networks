# This tool is for comparing two existing testing criteria for DNN models.

To compare two testing adequacy metrics, we used [Surprise Adequacy](https://github.com/coinse/sadl) and [Deep Mutation](https://github.com/dgl-prc/m_testing_adversatial_sample) as the two metrics, called Likelihood-based Surprise Coverage(LSC) and Label Change Rate(LCR).

In the following, procedure of each tool and nessesary changes is described:


## Testing adversarial Procedure:
We used [Deep Mutation](https://github.com/dgl-prc/m_testing_adversatial_sample) code to generate adversarial samples, mutation models and calculate label change rate. It is implemented in `Python2.7`

After generating adversarial examples, `mix_data.py` generates 10 mixed datasets of adversarial samples and normal samples.

Run `autorun.sh` to get the label change rate of the 10 datasets

Run `final_result.py` to get the result of increment of original data for the label change rate(LCR) metric.


## Surprise Coverage Procedure:
We used [Surprise Adequacy](https://github.com/coinse/sadl) code to compute surprise coverage, which is implemented in `Python2.7`, but we did some chanes to implement it in `Python3.7`.

Run `train_lenet5_model.ipynb`, `train_model.py`, `train_googlenet.ipynb` to train and save the deep learning model.

Run `img_to_npy.ipynb` to transfer the data to array format.

Run `sa_mnist_conv5.ipynb` to get the results of 10 datasets.

Run `final_result.py` to compute the increment of surprise coverage to the original data.



	

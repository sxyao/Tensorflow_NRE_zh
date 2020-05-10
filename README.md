# Chinese Neural Relation Extraction(CNRE)

In this project, we implement a combined word-level and sentence-level Bidirectional GRU network (BGRU+2ATT) with distant supervision for Chinese relation extraction from text. This project was inspired by project [TensorFlow-NRE](https://github.com/frankxu2004/TensorFlow-NRE).


# Data
We use the dataset provided by [TAC 2017 Cold Start KB Track](https://tac.nist.gov/2017/KBP/ColdStart/index.html).

Pre-Trained Word Vectors are learned from New York Times Annotated Corpus (LDC Data LDC2008T19), which should be obtained from LDC (https://catalog.ldc.upenn.edu/LDC2008T19). And we provide it in the origin_data/ directory.

Before you train your model, you need to run scripts from code/initialize/ to transform the original data into .npy files for the input of the network. The .npy files will be saved in data/ directory.

# Codes
The source codes are in the current main directory code/.

# Requirements
- Python (>=2.7)
- TensorFlow (=r0.11)
- scikit-learn (>=0.18)
- Matplotlib (>=2.0.0)

# Train
For training, you need to type the following command:  
`python train_GRU.py`  
The training model file will be saved in folder model/

You can lauch the tensorboard to see the softmax_loss, l2_loss and final_loss curve by typing the following command:
`tensorboard --logdir=./train_loss`  

# Test
For testing, you need to run the `test_GRU.py` to get all results on test dataset. BUT before you run it, you should change the pathname and modeliters you want to perform testing on in the test_GRU.py. We have add 'ATTENTION' to the code in `test_GRU.py` where you have to change before you test your own models.  

As an example, we provide our best model in the model/ directory. You just need to type the following command:  
`python test_GRU.py`  
The testing results will be printed(mainly the P@N results and the area of PR curve) and the all results on test dataset will be saved in out/ directory with the prefix "sample"  

To draw the PR curve for the sample model, you just need to type the following command:  
`python plot_pr.py`  
The PR curve will be saved as .png in current directory. If you want to plot the PR curve for your own model, you just need to change the modeliters in the `plot_pr.py` where we annotated 'ATTENTION'.



# Reference
[Zeng et al., 2014] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. Relation classification via convolutional deep neural network. In Proceedings of COLING.  

[Zeng et al.,2015] Daojian Zeng,Kang Liu,Yubo Chen,and Jun Zhao. Distant supervision for relation extraction via piecewise convolutional neural networks. In Proceedings of EMNLP.  

[Zhou et al.,2016] Zhou P, Shi W, Tian J, et al. Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification[C] Meeting of the Association for Computational Linguistics. 2016:207-212.  

[Lin et al., 2016] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. In Proceedings of ACL.

# CogniSNN
The repository includes code and supplementary material for  the paper "CogniSNN: An Exploration to Random Graph Architecture based Spiking Neural Networks with Enhanced Depth-Scalability and Path-Plasticity "

https://arxiv.org/abs/2505.05992

For classification experiments, taking DVS-Gesture as an example:

You can train the model in 
CogniSNN/ClassificationTask/DVS128Gesture/train.py

The code can be run directly, and all parameters are given.

In smodel.py, the class 'model' represents the 'CogniSNN' in paper, the class 'Randwire' represents 'RGA-based SNN' in paper, the class 'SEWblock' represents the ResNode in paper.


For continual learning experiments, taking CogniSNN/ContinualLearning/RGA-SNN-WS as an example:

Firstly, you must use the train.py, to train a model with old task(CIFAR100), as old model, its weight name Origin_Net.pth. 

Then you can use lwf.py to finish continual learning. 

In lwf.py, the lines 355-356 represent the node and edge in the path with the lowest betweenness centrality.   The lines 351-353 represent the node and edge in the path with the highest betweenness centrality. 

How to calculate the betweenness centrality? Just use the /CogniSNN/ContinualLearning/calculate_critical_path.py.  

To be honest, due to the time constrains, the code for continual learning is not very well written. I will continue to improve it in the future expansion work of CogniSNN.  If it causes inconvenience, please understand.

If you have any questions, please feel free to contact me through my email: Huangys124@163.com. Of course, you should ensure that this question is valuable.

If this article or repository is helpful to you, please cite it as follows (this is very important for me):

@article{huang2025cognisnn,
  title={CogniSNN: A First Exploration to Random Graph Architecture based Spiking Neural Networks with Enhanced Expandability and Neuroplasticity},
  author={Huang, Yongsheng and Duan, Peibo and Liu, Zhipeng and Sun, Kai and Zhang, Changsheng and Zhang, Bin and Xu, Mingkun},
  journal={arXiv preprint arXiv:2505.05992},
  year={2025}
}




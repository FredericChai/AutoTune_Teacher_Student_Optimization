# AutoTune_Teacher_Student_Optimization
This repo implement the teacher student network for optimization. It use the basian optmization to tune the hyperparameters in the method.

We use knowledge distillation (KD) [5] to transfer the knowledge in a pre-trained model (on general images) into a new model that is relevant for medical image domains. That means KD transfers knowledge from a complex model (teacher) to a simpler model (student). In the simplest form of distillation, KD uses a soft target distribution (i.e., soft label) from the complex model coupled with correct label (i.e., hard label) to train the student model. 

## Requirements
- [Scikit-learn](http://scikit-learn.org/stable/)
- [Pytorch](https://pytorch.org/) (Recommended version 9.2)
- [Python 3](https://www.python.org/)

## Quick Start
you can run the code in receipes to have a quick start. The main file requires two important hyper-parameters: α,T.
1.α is used to control the balance between the ground-truth loss and the filtered soft label loss. 
2.T can make the assistant loss will pay more attention on the classes with higher logits.

Belowing figure describe our method:

![image](https://github.com/FredericChai/AutoTune_Teacher_Student_Optimization/blob/main/Picture1.png)

We also implement the method on skin lesion segmentation：

![image](https://github.com/FredericChai/AutoTune_Teacher_Student_Optimization/blob/main/Picture2.png)

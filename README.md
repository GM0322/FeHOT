# Fouier-enhance High-Order Total Variation (FeHOT) Itrative Network for Interior Tomography

## Preparation
* Prepare python3 environment and install follow package
* pythorch
* numpy==1.22.4
* os
* astra
* CUDA
* odl=0.7.0

## Evaluation Demo
**We provide a trained model for demonstration**

First, Unzip the saved model file in the './checkpoints' folder.
Then, set parameters and run the following code:

```Shell
python validation.py
```

## trainning 
If you want re-train the network parameters, you can train as the following command. 
```Shell
python train.py
```

## Contact
For any question, please file an issue or contact
```
Genwei Ma: magenwei@126.com
```

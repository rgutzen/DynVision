## Methods
1) [x] Architecture schematic of the DyRCNN model family, native to the DynVision toolbox. [Schematic of layer and network]
2) [x] Dynamical systems formulation of the neural network activity with heterogenous connectivity and delays. [Equations]
3) [x] Network with recurrent, skip, and feedback connections unrolled in engineering versus biological time. [Schematic with engineering and biological unrolling]
4) [ ] types of kernel convolutions used to effectively realize recurrence. [Schematic of convolution connection types, incl. topographic]

## Results
5) [x] Training the DyRCNNx8 model with different recurrence types. [Plot with accuracy over training, and over time, losses, model stats]
    - [x] train/val accuracy / epoch, 
    - [x] losses / epoch, 
    - [x] stats: n params, max train acc, max val acc, avg gpu mem alloc, time/epoch
    - [x] Accuracy/confidence over time for one duration test [Dynamics plot]
    - [x] example dynamics of V1 [Dynamics plot]

9) [x] Tripytch Timeparams: tsteps, lossrt, idle
9) [x] Tripytch Connections: rctarget, skip, feedback
9) [x] Tripytch Timedelays: tau, trc, tsk

6) [x] Temporal RCNN dynamics for different recurrence types in response input with varied duration. [Dynamics plots and comparison to experimental data]
7) [x] Temporal RCNN dynamics for different recurrence types in response input with varied contrast. [Dynamics plots and comparison to experimental data]
8) [x] Temporal RCNN dynamics for different recurrence types in response input with varied delay interval. [Dynamics plots and comparison to experimental data]

14) [ ] noise robustness 
    - representative example, + ffonly

15) (imagenet training) don't delay because of this

## Supplement
13) [ ] Long-term stability plot 
11) [x] Equivalence of biological and engineering unrolling [Dynamics plots]
12) [ ] pure feedforward dynamics of recurrence-trained model [Dynamics plots]

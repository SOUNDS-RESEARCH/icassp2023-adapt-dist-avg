# Distributed Adaptive Norm Estimation For Blind System Identification in Wireless Sensor Networks

Distributed signal-processing algorithms in (wireless) sensor networks often aim to decentralize processing tasks to reduce communication cost and computational complexity or avoid reliance on a single device (i.e., fusion center) for processing.
In this contribution, we extend a distributed adaptive algorithm for blind system identification that relies on the estimation of a stacked network-wide consensus vector at each node, the computation of which requires either broadcasting or relaying of node-specific values (i.e., local vector norms) to all other nodes.
The extended algorithm employs a distributed-averaging-based scheme to estimate the network-wide consensus norm value by only using the local vector norm provided by neighboring sensor nodes.
We introduce an adaptive mixing factor between instantaneous and recursive estimates of these norms for adaptivity in a time-varying system.
Simulation results show that the extension provides estimation results close to the optimal fully-connected-network or broadcasting case while reducing inter-node transmission significantly.


## Repository content
This repository contains all code used to generate the plots for the paper submission.
- Python simulation code
## Instructions
In order to run the simulation, do the following:
- clone repository: ```git clone https://github.com/SOUNDS-RESEARCH/icassp2023-adapt-dist-avg.git```
- init submodule: ```git submodule update --init --recursive```
- create virtual python environment [optional]
- install dependencies: ```pip install -r requirements.txt```
- run simulations: ```python simulations/static.py``` and ```python simulations/dynamic.py```

## SOUNDS
This research work was carried out at the ESAT Laboratory of KU Leuven, in the frame of the SOUNDS European Training Network.

[SOUNDS Website](https://www.sounds-etn.eu/)

## Acknowledgements
<table>
    <tr>
        <td width="75">
        <img src="https://www.sounds-etn.eu/wp-content/uploads/2021/01/Screenshot-2021-01-07-at-16.50.22-600x400.png"  align="left"/>
        </td>
        <td>
        This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 956369
        </td>
    </tr>
</table>

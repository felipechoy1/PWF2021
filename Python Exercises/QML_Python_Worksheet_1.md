```python
# Filename: QML_Python_Worksheet_1.md
# Authors: Subadra Echeverria, Felipe Ixcamparic, M. Cerezo
# Date: November 29, 2021
# Description: QML_Python_Worksheet_1.ipynb visualization. Teaching porpuse only
```

# QML Python Worksheet

by M. Cerezo, Felipe Ixcamparic, Subadra Echeverria

This set of exercises is aimed at introducing the basic tools that will be later used for a Quantum Machine Learning (QML) implementation. 

We refer the reader to the [Qiskit webpage](https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html) for further details.

## Getting started

First thing's first. Lets install some packages we will use.


```python
!pip install qiskit
!pip install pylatexenc
```


```python
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
import qiskit.quantum_info as qi
from qiskit.quantum_info import *
from IPython.display import Image
from IPython.core.display import HTML 
```

## Hello World of Quantum Circuits

Here we will show how to use the basics of Qiskit. 


```python
# Create a Quantum Circuit acting on a quantum register of two qubits
qcirc = QuantumCircuit(2)
```

We want to write the circuit that prepares a Bell pair.


```python
# Add a H gate on qubit 0, putting this qubit in superposition.
qcirc.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qcirc.cx(0, 1)
```




    <qiskit.circuit.instructionset.InstructionSet at 0x7f23bc2289a0>




```python
# We can visualize the circuit
qcirc.draw()
```




![png](output_9_0.png)



Qiskit has a function called `statevector_simulator' which prints the quantum state obtained at the output of the circuit.


```python
# Run the quantum circuit on a statevector simulator backend
backend = Aer.get_backend('statevector_simulator')

# Create a Quantum Program for execution
job = backend.run(qcirc)
result = job.result()
outputstate = result.get_statevector(qcirc, decimals=10)
print(outputstate)
```

    [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]


We can also simulate quantum measurements. For this purpose, we need to add measurements to the original circuit above, and use a different Aer backend.


```python
# Create a Quantum Circuit with 2 quantum registers and 2 classical registers
meas = QuantumCircuit(2, 2)
meas.barrier(range(2))
# map the quantum measurement to the classical bits
meas.measure(range(2), range(2))

# The Qiskit circuit object supports composition using the addition operator.
qcirc.add_register(meas.cregs[0])
qc = qcirc.compose(meas)

#drawing the circuit
qc.draw()
```




![png](output_13_0.png)




```python
# Use Aer's qasm_simulator
backend_sim = Aer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator. We've set the number of repeats of the circuit to be 1024, which is the default.
N_shots=1024
job_sim = backend_sim.run(transpile(qc, backend_sim), shots=N_shots)

# Grab the results from the job.
result_sim = job_sim.result()
```


```python
# We can print the measurement outcomes and the probability of each outcome
counts = result_sim.get_counts(qc)
print(counts)
probability={}
for ele in counts:
  probability[ele]=counts[ele]/N_shots
print(probability)
```

    {'11': 500, '00': 524}
    {'11': 0.48828125, '00': 0.51171875}



```python
# And we can visualize them
plot_histogram(counts)
```




![png](output_16_0.png)



**Note**: This representation of the bitstring puts the most significant bit (MSB) in the left, and the least significant bit (LSB) on the right. This is the standard ordering of binary bitstrings. We order the qubits in the same way (qubit representing the MSB has index 0), which is why Qiskit uses a non-standard tensor product order.

We can double-check that the error in those probabilities are within the statistical uncertainty of order <img src = "https://render.githubusercontent.com/render/math?math=1/\sqrt{N}">. Here <img src = "https://render.githubusercontent.com/render/math?math=N=1024"> . We recall that this uncertainty tells us that <img src = "https://render.githubusercontent.com/render/math?math=p(i)=N_i/N\pm 1/\sqrt{N}">, where <img src = "https://render.githubusercontent.com/render/math?math=N_i"> is the probability of the <img src = "https://render.githubusercontent.com/render/math?math=\large i">-th outcome.


```python
statistical_error=1/N_shots**.5
print(statistical_error)
```

    0.03125


We can also check that the measurement outcomes are correct by using the  "statevector_simulator" and computing <img src="https://render.githubusercontent.com/render/math?math=p(00)%3D%5Ctext%7BTr%7D%5BM_%7B00%7D%20%5Crho%5D">, with  <img src = "https://render.githubusercontent.com/render/math?math=\rho=\left|\psi\rangle\langle\psi\right|">.

Here, we use 

<img src="https://render.githubusercontent.com/render/math?math=%24M_%7B00%7D%3D%5Cleft%7C0%5Crangle%5Clangle%200%5Cright%7C%5Cotimes%5Cleft%7C0%5Crangle%5Clangle%200%5Cright%7C%3D%5Cbegin%7Bpmatrix%7D1%260%5C%5C0%260%5Cend%7Bpmatrix%7D%5Cotimes%20%5Cbegin%7Bpmatrix%7D1%260%5C%5C0%260%5Cend%7Bpmatrix%7D%3D%0A%5Cbegin%7Bpmatrix%7D1%20%26%200%20%26%200%20%26%200%5C%5C%0A0%20%26%200%20%26%200%20%26%200%5C%5C%0A0%20%26%200%20%26%200%20%26%200%5C%5C%0A0%20%26%200%20%26%200%20%26%200%0A%5Cend%7Bpmatrix%7D%0A%5Cbegin%7Balign*%7D%5Cend%7Balign*%7D%24">


```python
M00=np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
rho=np.outer(outputstate, outputstate.conj())
p00=np.trace(np.matmul(M00,rho))
print(p00)
```

    (0.5000000000190247+0j)


We can also create a circuit where we only measure one qubit.



```python
# Create a Quantum Circuit acting on a quantum register of two qubits
qcirc2 = QuantumCircuit(2)
qcirc2.h(0)
qcirc2.cx(0, 1)
qcirc2.draw()
```




![png](output_23_0.png)




```python
# Create a Quantum Circuit with 2 quantum registers and 1 classical registers
meas2 = QuantumCircuit(2, 1)
meas2.barrier(range(2))
# map the quantum measurement to the classical bits
meas2.measure(0, 0)

# The Qiskit circuit object supports composition using the addition operator.
qcirc2.add_register(meas2.cregs[0])
qc2 = qcirc2.compose(meas2)

#drawing the circuit
qc2.draw()
```




![png](output_24_0.png)



These are the measurement outcomes.


```python
# Use Aer's qasm_simulator
backend_sim = Aer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator. We've set the number of repeats of the circuit to be 1024, which is the default.
N_shots=1024
job_sim = backend_sim.run(transpile(qc2, backend_sim), shots=N_shots)

# Grab the results from the job.
result_sim = job_sim.result()

# We can print the measurement outcomes and the probability of each outcome
counts = result_sim.get_counts(qc2)
print(counts)
probability={}
for ele in counts:
  probability[ele]=counts[ele]/N_shots
print(probability)
```

    {'0': 500, '1': 524}
    {'0': 0.48828125, '1': 0.51171875}



```python
# And we can visualize them
plot_histogram(counts)
```




![png](output_27_0.png)



We can double check that the measurement outcomes are correct by using the  "statevector_simulator" and computing  <img src = "https://render.githubusercontent.com/render/math?math=p_A(0)=\text{Tr}[(M_{0}\otimes I_B) \rho]">, with <img src = "https://render.githubusercontent.com/render/math?math=\rho=\left|\psi\rangle\langle\psi\right|">. 

Note that due to qiskit's qubit ordering convention, we need to build the kronecker product with different qubit ordering.

Here, we use  <img src="https://render.githubusercontent.com/render/math?math=%24(M_%7B0%7D%5Cotimes%20I_B)%3D%5Cbegin%7Bpmatrix%7D1%260%5C%5C0%261%5Cend%7Bpmatrix%7D%5Cotimes%5Cbegin%7Bpmatrix%7D1%260%5C%5C0%260%5Cend%7Bpmatrix%7D%3D%0A%5Cbegin%7Bpmatrix%7D1%260%260%260%5C%5C0%260%260%260%5C%5C0%260%261%260%5C%5C0%260%260%260%5Cend%7Bpmatrix%7D%0A%5Cbegin%7Balign*%7D%5Cend%7Balign*%7D%24">



```python
M00=np.array([[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
rho=np.outer(outputstate, outputstate.conj())
p00=np.trace(np.matmul(M00,rho))
print(p00)
```

    (0.5000000000190247+0j)


## Quantum Operations

You can find a summary of Qiskit's quantum operations [here](https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/circuits/3_summary_of_quantum_operations.ipynb).

For now, let us create a single qubit state that is rotated about the <img src = "https://render.githubusercontent.com/render/math?math=Y">-axis on an angle  <img src = "https://render.githubusercontent.com/render/math?math=\theta">.


```python
def rotation_Y(quantum_circuit,theta)-> None:
  '''
    Applying a rotation about the Y axis to a single qubit

    Parameters
    ----------
    quantum_circuit: QuantumCircuit
      The quantum cirquit to which we apply ansatz
    theta: float
        The parameters for the R_y gate

  '''
  quantum_circuit.ry(theta,0)
```


```python
# We set theta=pi/2
qcirc3 = QuantumCircuit(1)
rotation_Y(qcirc3,np.pi/2)
qcirc3.draw()
```




![png](output_32_0.png)



We denote as <img src="https://render.githubusercontent.com/render/math?math=%24%5Cleft%7C%5Cpsi(%5Ctheta)%5Cright.%5Crangle%3D%5Clarge%20R_y(%5Ctheta)%5Cleft%7C0%5Crangle%5Cright.%24">
 the otput state from the previous circuit.

### Computing expectation values.

Here we will compute the expectation value of the  <img src = "https://render.githubusercontent.com/render/math?math=Z"> operator over a single qubit quantum state.


```python
# First, let us define the Z-Pauli operator
Z=np.array([[1,0],[0,1]])
print(Z)

#We can also import it from qiskit as 



print(qi.Pauli('Z').to_matrix())
```

    [[1 0]
     [0 1]]
    [[ 1.+0.j  0.+0.j]
     [ 0.+0.j -1.+0.j]]


**Exercice 1:** 

Use the statevector_simulator backend to compute the expectation value of <img src = "https://render.githubusercontent.com/render/math?math=Z"> as a function of <img src = "https://render.githubusercontent.com/render/math?math=\theta">. Here, use  matrix multiplication to compute <img src = "https://render.githubusercontent.com/render/math?math=\langle Z\rangle=\langle\psi(\theta)\left|Z\right|\psi(\theta)\rangle">. Make a plot of <img src = "https://render.githubusercontent.com/render/math?math=\langle Z\rangle"> versus <img src = "https://render.githubusercontent.com/render/math?math=\theta">.


```python
# Solution to Exercice 1:
```

**Exercice 2:**

Simulate measurements to compute the expectation value of <img src = "https://render.githubusercontent.com/render/math?math=Z"> as a function of <img src = "https://render.githubusercontent.com/render/math?math=\theta">.


```python
# Solution to Exercice 2:
```

## First Variational Quantum Algorithm

From the previous, we see that it is very easy to verify that 
<img src = "https://render.githubusercontent.com/render/math?math=\langle Z\rangle"> is minimized for <img src = "https://render.githubusercontent.com/render/math?math=\theta=-\pi">. However, we want to solve this problem numerically. First, we load an optimizer.


```python
# Load an optimization package
from scipy.optimize import minimize
```

**Exercice 3:** 
Define a cost function <img src = "https://render.githubusercontent.com/render/math?math=C(\theta)"> that takes as input a parameter <img src = "https://render.githubusercontent.com/render/math?math=\theta"> and outputs <img src = "https://render.githubusercontent.com/render/math?math=\langle\psi(\theta)\left|Z\right|\psi(\theta)\rangle">


```python
# Solution to Exercice 3:
def cost_function(theta):
  '''
  Define here.
  '''
  return(cost_value)

```


```python
## We can optimize the cost function using the minimize function.

# First, we set a random parameter to start the optimziation
t0 = np.random.rand() * 2 * np.pi

# We then run the optimization using the COBYLA method, with a maximum of 300 iteration steps
out = minimize(cost_function, x0, method="COBYLA", options={'maxiter':300}, callback=callback)
```

**Exercice 4:** 
Verify that the optimization finds the correct solution of 
<img src = "https://render.githubusercontent.com/render/math?math=\theta=\pi\,\,\text{mod}(2\pi)">

**Exercice 5:** 
Plot the cost function value versus iteration step.

Tip: define an empty array `cost_function_values = [ ]` and add a line to the cost function that appends cost value to the previous array: `cost_function_values.append( )`.


```python
# Solution to Exercice 5:
```

## Second Variational Quantum Algorithm

Here we implement a variational algorithm in two qubits.


```python
# First, let us define a two qubit gate that we will use as a basis for the parametrized two qubit circuit.

def two_qubit_gate(quantum_circuit,parameters,q0,q1)-> None:
  '''
    Applying a rotation about the Y axis to a single qubit

    Parameters
    ----------
    quantum_circuit: QuantumCircuit
      The quantum cirquit to which we apply ansatz
    parameters: float
        The parameters for the rotations 
    q1: int
      first qubit that the gate acts on
    q2: int
      second qubit that the gate acts on

  '''
  quantum_circuit.ry(parameters[0],q0)
  quantum_circuit.ry(parameters[1],q1)
  quantum_circuit.cx(q0,q1)
  quantum_circuit.ry(parameters[2],q0)
  quantum_circuit.ry(parameters[3],q1)

```


```python
# Here we show the action of the two-qubit gate with 4 random angles
random_parameters = np.random.rand(4) * 2 * np.pi
qcirc4 = QuantumCircuit(2)
two_qubit_gate(qcirc4,random_parameters,0,1)
qcirc4.draw()
```




![png](output_50_0.png)



**Exercice 6:** 
If we apply two layers of the `two_qubit_gate unitary` we will have a redundancy of two Ry gates acting one after the other. How can we remove this parameter redundancy?


```python
# Solution to Exercice 6:
```

**Exercice 7:**

Write cost functions that prepare states that minimize the expectation values <img src = "https://render.githubusercontent.com/render/math?math=\langle Z\otimes Z\rangle"> and <img src = "https://render.githubusercontent.com/render/math?math=\langle X\otimes I\rangle"> 


```python
# Solution to Exercice 7:
```

## The Ising Model
The Ising model is a well-known condensed matter model that describes ferromagnetism in statistical mechanics. At its core, it describes interactions between spins in the systems and the interactions between the spins and a magnetic field along.

The Hamiltonian of the model is

<img src="https://render.githubusercontent.com/render/math?math=H%3D-%5Csum_%7Bi%3D1%7D%5En%20X_%7Bi%7D%5Cotimes%20X_%7Bi%5C%2B1%7D-g%5Csum_%7Bi%3D1%7D%5En%20Z_i">.

Here, the terms <img src="https://render.githubusercontent.com/render/math?math=X_i%5Cotimes%20X_%7Bi%2B1%7D"> describe the spin-spin interaction, while <img src = "https://render.githubusercontent.com/render/math?math=Z_i"> is the interaction between the spins and a magnetic field along the <img src = "https://render.githubusercontent.com/render/math?math=z"> direction. 

This ground states of this model have a `phase transition` at <img src = "https://render.githubusercontent.com/render/math?math=g=1"> , such that for <img src = "https://render.githubusercontent.com/render/math?math=g\leq 1"> the states are paramegnetic, while for 
<img src = "https://render.githubusercontent.com/render/math?math=g\geq 1"> they are ferromagnetic.

**Exercice 8:**

Explicitly write the matrix for the Hamiltonian <img src = "https://render.githubusercontent.com/render/math?math=H">  of the Ising model for <img src = "https://render.githubusercontent.com/render/math?math=n=2,4,6"> spins. Then, make a plot of the eigenvalues of <img src = "https://render.githubusercontent.com/render/math?math=H"> as a function of <img src = "https://render.githubusercontent.com/render/math?math=g">. What happens at <img src = "https://render.githubusercontent.com/render/math?math=g=1">?


```python
# Solution to Exercice 8:
```

**Exercice 9:**

Create a Variational Quantum Eigensolver (VQE) algorithm that finds the ground state of <img src = "https://render.githubusercontent.com/render/math?math=H"> for <img src = "https://render.githubusercontent.com/render/math?math=g=.25,.5,1.25,1.5"> (for <img src = "https://render.githubusercontent.com/render/math?math=n=4"> or <img src = "https://render.githubusercontent.com/render/math?math=n=6">).

For this purpose, first create a layered ansatz as the one shown bellow.


```python
Image(url= "https://entangledphysics.files.wordpress.com/2021/11/ansatz.png")
```




<img src="https://entangledphysics.files.wordpress.com/2021/11/ansatz.png"/>



Here, <img src = "https://render.githubusercontent.com/render/math?math=\large R_y"> are rotations about the <img src = "https://render.githubusercontent.com/render/math?math=\large y"> axis (with independent angles). We show an ansatz with 3 layers.


```python
# Solution to Exercice 9:
```

## A Quantum Dataset
The Ising model provides a perfect way to create a dataset for quantum machine learning. Specifically, we will here create a supervised learning task of clasfifying states according go the phase of matter they belong to.

As previously mentioned, the ground states of the Ising model are paramagnetic for <img src = "https://render.githubusercontent.com/render/math?math=g\leq 1">, and ferromagnetic for <img src = "https://render.githubusercontent.com/render/math?math=g\geq 1">.

As shown in the figure below, this allows us to create a dataset of ground states <img src = "https://render.githubusercontent.com/render/math?math=\left|\psi_i\rangle\right."> belonging to the either the paramagnetic of ferromagnetic phase.


```python
Image(url="https://entangledphysics.files.wordpress.com/2021/11/qml.png")
```




<img src="https://entangledphysics.files.wordpress.com/2021/11/qml.png"/>



**Exercice 10:**

Randomly sample 30 fields <img src = "https://render.githubusercontent.com/render/math?math=g"> in each phase and run VQE algorithms to create a dataset of <img src = "https://render.githubusercontent.com/render/math?math=60"> ground states of the Ising model.

We will divide this data into 2 sets:

A training set of <img src = "https://render.githubusercontent.com/render/math?math=40"> states, and a testing set of <img src = "https://render.githubusercontent.com/render/math?math=20"> states.


```python
# Solution to Exercice 10:
```


Once we have the dataset, we can send the states trough a Quantum Neural Network that will learn to classify them. In this case, we will use a Quantum Convolutional Neural Network (QCNN). Below we show a QCNN for 4 qubits.


```python
Image(url="https://entangledphysics.files.wordpress.com/2021/11/qcnn.png")
```




<img src="https://entangledphysics.files.wordpress.com/2021/11/qcnn.png"/>



**Exercice 11:**

Write a code for the QCNN ansatz.


```python
# Solution to Exercice 11:
```

The way in which we assign labels is by measuring the expectation value of the Pauli <img src = "https://render.githubusercontent.com/render/math?math=Z"> operator on the output single-qubit state of the QCNN <img src = "https://render.githubusercontent.com/render/math?math=\rho_i(\alpha)">, where <img src = "https://render.githubusercontent.com/render/math?math=\alpha"> are the trainable parameters in the QCNN.

That is, the assigned label is <img src = "https://render.githubusercontent.com/render/math?math=\widetilde{y}_i(\alpha)=\text{Tr}[\rho_i(\alpha)Z]">.

Note that, one can always compute this expectation value (using qiskit inverse kronecker product notation) as 

<img src = "https://render.githubusercontent.com/render/math?math=\widetilde{y}_i(\alpha)=\text{Tr}[\left|\psi_i(\alpha)\right.\rangle\langle \psi_i(\alpha)\left|(Z\otimes I\otimes I\otimes I)\right.]=\langle \psi_i(\alpha)\left|(Z\otimes I\otimes I\otimes I)\right|\psi_i(\alpha)\rangle">.



And we train the mean-squared error loss function:

<img src = "https://render.githubusercontent.com/render/math?math=\mathcal{L}(\alpha)=\frac{1}{40}\sum_{i=1}^{40}(\widetilde{y}_i(\alpha)-y_i)^2">.

**Exercice 12:**

Train the QCNN to classify quantum states according to the phase of matter that they belong to.


```python
# Solution to Exercice 12:
```

Once we have trained the QCNN, we need to test its accurancy in clasifying. Now, evidently, the expectation values previously computed are continous numbers in <img src = "https://render.githubusercontent.com/render/math?math=[-1,1]">, while the true labels are discrete values in the set <img src = "https://render.githubusercontent.com/render/math?math=\{-1,1\}">.

In order to assing a discrete label to each state, we need a discretizing function. For instance, this can be achieved by

<img src = "https://render.githubusercontent.com/render/math?math=\widehat{y}_i=1"> if <img src = "https://render.githubusercontent.com/render/math?math=\widetilde{y}_i(\alpha)\geq 0">, 

and 

<img src = "https://render.githubusercontent.com/render/math?math=\widehat{y}_i=-1"> if <img src = "https://render.githubusercontent.com/render/math?math=\widetilde{y}_i(\alpha)< 0">.

Given this discretization, we can now use the <img src = "https://render.githubusercontent.com/render/math?math=20"> states in the testing set  to check how many of the assigned labels <img src = "https://render.githubusercontent.com/render/math?math=\widehat{y}_i"> match with the true labels <img src = "https://render.githubusercontent.com/render/math?math=y_i">.

**Exercice 13:**

Compute the percentage of correct label assignment on the testing set.


```python
# Solution to Exercice 13:
```

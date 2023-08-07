# System Identification for Temporal Networks

<p>Modelling the dynamics of temporal networks as a linear process
using a system identification approach. </p>
Authors: Sergey Shvydun and Piet Van Mieghem

## Generate Periodic Sequence
    >>> from graph_generator import *
    >>> N, T= 10, 200   # N - number of nodes, T - timestamps
    >>> u, y =  graph_simple_sequence(N, T, 1, 1)   # Generate periodic graph sequence

## Example: SG-gen model

Get Q and x[0] by an SG-gen model: 

    >>> import si
    >>> q, x0 = si.sg_gen(u,y)


## Example: LPG-gen model

Generate a network using NetworkX package 

    >>> import matrix_identification
    >>> q, x0 = matrix_identification.lpg_gen(u,y)

## Example: LG-gen model

Generate a network using NetworkX package 


    >>> import pt
    >>> import graph_generator as gg
    >>> N, perionds, changes= 10, 4, 8   # N - number of nodes, T - timestamps
    >>> u, y = gg.generate_noise_data2(N, perionds, changes)
    >>> out = pt.lg_gen(u, 5, modelling=True)

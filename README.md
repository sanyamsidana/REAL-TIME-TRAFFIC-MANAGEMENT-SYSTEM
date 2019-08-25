REAL TIME TRAFFIC MANAGEMENT SYSTEM 

Due to the emerging problem of traffic congestion nowadays,there are many problems
from which the human race is suffering and is affected too. In order to bring some control
over these situations we came forward with the idea of traffic control using image
processing and machine learning. Tracking the moving vehicles using image can help us in
achieving quantitative description of traffic flow.

In the following project we tried to solve the problem of traffic congestion using image
processing. We take an image of a congested road from the traffic signal point of view. We
then determine the density of that particular road using openCV library. After we have the
density of the congested road,we then move forward to use machine learning in order to
solve the congestion problem.

Liner Regression is used in order to determine the amount of cars from the density output
we achieved from openCV. After the amount of cars is determined, again linear regression is
used to determine the time required for that amount of cars in the congested road to move.

We also implemented another algorithm in order to manage the timer properly so that no
other road gets congested due to the ongoing green signal of a road. Our implementation
also keeps in notice that no other road is heavily congested during green signal of another
road. A threshold of timers is maintained for functioning of non congestion of all roads efficiently.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:47:04 2026

@author: tyler
"""

import math

class Vector:
    
    def __init__(self, elements):
        self.elements = tuple(elements)
        self.magnitude = self.calc_magnitude()
        return
    
    def calc_magnitude(self):
        '''
        Calculate Vector magnitude.

        Returns
        -------
        m : float
            Magnitude of Vector.

        '''
        m = sum([e**2 for e in self])
        m = m ** (1/2)
        return m
    
    def unit(self):
        '''
        Calculate unit Vector parallel to self.

        Returns
        -------
        u : Vector
            Unit-length Vector parallel to self.

        '''
        u = self/self.magnitude
        return u
    
    def rotate(self, angle, axis):
        '''
        Rotate Vector counterclockwise around another vector by the specified
        angle. Utilizes Rodrigues' rotation formula.

        Parameters
        ----------
        angle : float
            Angle of rotation, in degrees.
        axis : Vector
            Axis about which the rotation will be performed.

        Returns
        -------
        rotated : Vector
            Rotated vector.
        '''
        
        def deg2rad(deg):
            rad = (deg / 360) * 2 * math.pi
            return rad
    
        def rad2deg(rad):
            deg = (rad / 2 / math.pi) * 360
            return deg
        
        angle = deg2rad(angle)
        term1 = self * math.cos(angle)
        term2 = cross(axis, self) * math.sin(angle)
        term3 = axis * dot(axis, self) * (1 - math.cos(angle))
        rotated = term1 + term2 + term3
        return rotated
            
    def __iter__(self):
        for e in self.elements:
            yield e
        return
    
    def __str__(self):
        s = str(self.elements)
        return s
    
    def __repr__(self):
        s = str(self.elements)
        return s
    
    def __mul__(self, other):
        return Vector([e * other for e in self])

    def __truediv__(self, other):
        return Vector([e / other for e in self])
    
    def __add__(self, other):
        assert type(other) is Vector, 'Object being added is not a Vector'
        assert len(self) == len(other), 'Vectors being added must have the same length'
        return Vector([a+b for a,b in zip(self, other)])
    
    def __sub__(self, other):
        assert type(other) is Vector, 'Object being subtracted is not a Vector'
        assert len(self) == len(other), 'Vectors being subtracted must have the same length'
        return Vector([a-b for a,b in zip(self, other)])
        
    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, index):
        return self.elements[index]

class Simulation:
    
    def __init__(self):
        self.bodies = {}
        self.forces = {}
        self.time_step = 1
        self.steps = 0
        self.elapsed_time = 0
        self.paths = {}
        return
    
    def step(self):
        '''
        Perform one simulation step.

        Returns
        -------
        None.

        '''
        self.update_forces()
        self.update_velocities()
        self.update_positions()
        self.steps += 1
        self.elapsed_time += self.time_step
        for k, body in self.bodies.items():
            self.paths[k].append(body.position)
        return

    def add_body(self, body):
        '''
        Add a new body to the Simulation.

        Parameters
        ----------
        body : Body
            New Body to be added to Simulation.

        Returns
        -------
        None.

        '''
        keys = list(self.bodies.keys())
        if len(keys) == 0:
            i = 0
        else:
            i = max(keys) + 1
        self.bodies[i] = body
        self.paths[i] = [body.position]
        return
    
    def update_forces(self):
        '''
        Update forces acting on Simulation bodies.

        Returns
        -------
        forces : dict
            Net force Vectors acting on all bodies.

        '''
        
        forces = {}
        G = 6.6743015E-11   #m^3/kg s^2
        for i, body_i in self.bodies.items():
            net_force = Vector([0,0,0])
            for j, body_j in self.bodies.items():
                if i == j:
                    continue
                displacement = body_j.position - body_i.position
                magnitude = G * ((body_i.mass * body_j.mass)/(displacement.magnitude**2))
                net_force += displacement.unit() * magnitude
            forces[i] = net_force
        self.forces = forces
        return forces
    
    
    def update_velocities(self):
        '''
        Update velocities of all Simulation Bodies according to most recently
        calculated force vectors.

        Returns
        -------
        None.

        '''
        accelerations = {}
        for i, force in self.forces.items():
            mass = self.bodies[i].mass
            a = force / mass
            accelerations[i] = a
        for i, a in accelerations.items():
            self.bodies[i].velocity += a*self.time_step
        return
    
    def update_positions(self):
        '''
        Update positions of all Bodies according to current velocity Vectors.

        Returns
        -------
        None.

        '''
        for body in self.bodies.values():
            body.position += body.velocity * self.time_step
        return

    def center_of_mass(self):
        '''
        Calculate center-of-mass for all Bodies in Simulation.

        Returns
        -------
        com : Vector
            3-vector corresponding to calculated center-of-mass.

        '''
        com = center_of_mass(list(self.bodies.values()))
        return com

class Body:
    
    def __init__(self):
        self.position = None
        self.velocity = None
        self.mass = None
        return
        
    def set_position(self, x, y, z):
        '''
        Set position of Body in 3D space, in meters.

        Parameters
        ----------
        x : float
            x-coordinate of Body position.
        y : float
            y-coordinate of Body position.
        z : float
            z-coordinate of Body position.

        Returns
        -------
        None

        '''
        self.position = Vector((x,y,z))
        return
    
    def set_mass(self, mass):
        '''
        Set mass of Body, in kilograms.

        Parameters
        ----------
        mass : float
            Mass of Body, in kilograms.

        Returns
        -------
        None.

        '''
        self.mass = mass
        return
    
    def set_velocity(self, x, y, z):
        '''
        Set velocity Vector of Body, in meters per second.

        Parameters
        ----------
        x : float
            x-coordinate of Body velocity.
        y : float
            y-coordinate of Body velocity.
        z : float
            z-coordinate of Body velocity.

        Returns
        -------
        None.

        '''
        self.velocity = Vector((x, y, z))
        return
    
    def __iter__(self):
        for e in self.position:
            yield e
        return
    
# if __name__ == "__main__":
    
#     import matplotlib.pyplot as plt
    
#     sim = Simulation()
    
#     b1 = Body()
#     b1.set_position(0,100,0)
#     b1.set_mass(100)
#     b1.set_velocity(0.005,0,0)
#     sim.add_body(b1)
#     b2 = Body()
#     b2.set_position(0,350,0)
#     b2.set_mass(10000)
#     b2.set_velocity(0.004,-0.0031,0)
#     sim.add_body(b2)
#     b3 = Body()
#     b3.set_position(50,200,0)
#     b3.set_mass(100000000)
#     b3.set_velocity(-0.001,0,0.0001)
#     sim.add_body(b3)
    
#     trails = [[] for b in sim.bodies.items()]
#     for i, b in enumerate(sim.bodies.values()):
#         trails[i].append(list(b))
    
#     forces = sim.update_forces()
#     sim.update_velocities()
#     sim.update_positions()
#     for i, b in enumerate(sim.bodies.values()):
#         trails[i].append(list(b))
        
#     sim.step()
#     for i, b in enumerate(sim.bodies.values()):
#         trails[i].append(list(b))
    
#     for i in range(3600*24*7):
#         sim.step()
#         for i, b in enumerate(sim.bodies.values()):
#             trails[i].append(list(b))
            
#     plt.figure()
#     for j in range(len(sim.bodies)):
#         plt.scatter([t[0] for t in trails[j]], [t[1] for t in trails[j]], alpha=[i/sim.steps for i in range(sim.steps)], s=1)

def lagrange(body1, body2):
    '''
    Calculate Lagrange points for a pair of bodies.
    
    ||in-progress||

    Parameters
    ----------
    body1 : Body
        Body 1.
    body2 : Body
        Body 2.

    Returns
    -------
    l_points : tuple of Vectors
        tuple containing calculated Lagrange points (L1, L2, L3, L4, L5).

    '''
    #Assign major and minor bodies
    if body1.mass >= body2.mass:
        major = body1
        minor = body2
    else:
        major = body2
        minor = body1
    
    #Obtain vector from major to minor
    displacement = minor.position - major.position
    R = displacement.magnitude
    
    #Calculate L1
    u = minor.mass / (major.mass + minor.mass)
    r = R * (u/3)**(1/3)
    L1 = minor.position - (displacement.unit() * r)
    
    #Calculate L2
    L2 = minor.position + (displacement.unit() * r)
    
    r = R * (7/12) * u
    L3 = (displacement * -1) + displacement.unit() * r 
    
    #Calculate L4
    barycenter = center_of_mass([major, minor])
    minor_lookahead = minor.position + minor.velocity
    norm = cross(displacement, minor_lookahead - barycenter).unit()
    L4 = displacement.rotate(60, norm)
    
    #Calculate L5
    L5 = displacement.rotate(-60, norm)
    
    l_points = (L1, L2, L3, L4, L5)
    
    return l_points
        
def center_of_mass(bodies):
    '''
    Calculate center-of-mass between two or more bodies.

    Parameters
    ----------
    bodies : iterable
        Iterable group of Body objects.

    Returns
    -------
    com : Vector
        Center-of-mass of input Body objects.

    '''
    com = Vector((0,0,0))
    total_mass = 0
    for body in bodies:
        total_mass += body.mass
    for body in bodies:
        com += body.position * (body.mass/total_mass)
    return com

#Vector math
def dot(v1, v2):
    '''
    Calculate dot product between two vectors.

    Parameters
    ----------
    v1 : Vector
        Vector 1.
    v2 : Vector
        Vector 2.

    Returns
    -------
    d : float
        Dot product of v1 and v2.

    '''
    assert len(v1) == len(v2)
    d = 0
    for i in range(len(v1)):
        d += v1[i] * v2[i]
    return d

def angle(v1, v2):
    '''
    Calculate angle between two vectors.

    Parameters
    ----------
    v1 : Vector
        Vector 1.
    v2 : Vector
        Vector 2.

    Returns
    -------
    t : float
        Angle between v1 and v2, in radians.
    '''
    d = dot(v1, v2)
    p = v1.magnitude * v2.magnitude
    t = math.acos(d/p)
    return t

def sgp(body1, body2):
    '''
    Calculate standard gravitational parameter u of two bodies.

    Parameters
    ----------
    body1 : Body
        Body 1.
    body2 : Body
        Body 2.

    Returns
    -------
    u : float
        Standard gravitational parameter of body1-body2 system.
    '''
    
    G = 6.6743015E-11
    u = G * (body1.mass + body2.mass)
    return u

def cross(v1, v2):
    '''
    Calculate cross product of two 3-vectors.
    
    Parameters
    ----------
    v1 : Vector
        Vector 1.
    body2 : Vector
        Vector 2.

    Returns
    -------
    c : Vector
        Cross product of vectors v1 and v2.
    '''
    assert len(v1) == 3
    assert len(v2) == 3
    c = Vector((v1[1]*v2[2] - v1[2]*v2[1],
                v1[2]*v2[0] - v1[0]*v2[2],
                v1[0]*v2[1] - v1[1]*v2[0]))
    return c
    
    
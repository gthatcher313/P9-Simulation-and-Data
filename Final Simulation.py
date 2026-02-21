#Imports
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from astropy.time import Time
from scipy.stats import halfnorm
from scipy.optimize import newton
import time

starttime = time.time()
#Core Functions
G = c.G.value
def grav(p,m):
    nummassive = len(m)
    testparticles = p[nummassive:]
    p = p[:nummassive]
    accel = p*0
    testaccel = testparticles*0
    p_rel = p[:,None,:] - p[None,:,:]
    dist = np.linalg.norm(p_rel,axis=2)
    np.fill_diagonal(dist,np.inf)
    accel = np.sum(-G * m[None,:,None] * p_rel / dist[:, :, None]**3, axis = 1)    
    test_rel = testparticles[:,None,:] - p[None,:,:]
    testdist = np.linalg.norm(test_rel,axis=2)
    testaccel = np.sum(-G * m[None,:,None] * test_rel / testdist[:, :,None]**3, axis = 1)
    netaccel = np.vstack([accel,testaccel])
    return netaccel
"""
grav(p,m) calculates the gravitational acceleration on each body given the positions p (shape (N,3)) and masses m (shape (N,)).
The first bodies are considered massive and the rest are test particles. The function returns an array of shape (N,3) containing the acceleration on each body.
The function first calculates the pairwise relative positions and distances between the massive bodies to compute their mutual accelerations.
Then it calculates the relative positions and distances between the test particles and the massive bodies to compute the accelerations on the test particles using the same process. 
Finally, it combines these accelerations into a single array (shape (N,3)) and returns it.
"""


def energy(p,v,m):
    nummassive = len(m)
    ptemp = p[:nummassive]
    vtemp = v[:nummassive]
    vsquarebyobj= np.sum(vtemp**2, axis=1)
    KE = np.sum(vsquarebyobj*m*0.5)
    p_rel = ptemp[:, None, :] - ptemp[None, :, :]
    dist = np.linalg.norm(p_rel, axis=2)
    MxM = m[:, None] * m[None, :] 
    np.fill_diagonal(dist, np.inf)
    U_pairwise = -G * MxM / dist
    PE = np.sum(U_pairwise*0.5)
    return PE+KE
"""
energy(p,v,m) calculates the total energy of the system given the positions p (shape (N,3)), velocities v (shape (N,3)), and masses m (shape (N,)) of the bodies.
The function first separates the massive bodies from the test particles and calculates the kinetic energy (KE) of the massive bodies using their velocities and masses.
Then it calculates the pairwise potential energy (PE) between the massive bodies by computing their relative positions and distances
, and using the gravitational potential energy formula. The function sums up the kinetic and potential energy to return the total energy of the system.
"""

def orbitcalc(semimaj, eccentricity, inclination, trueanomaly, periapsis, longascending, mu=G*c.M_sun.value):
    # Convert angles to radians
    e = eccentricity
    Ω = longascending.to(u.rad).value
    a = semimaj.to(u.m).value
    ω = periapsis.to(u.rad).value
    i = inclination.to(u.rad).value
    v = trueanomaly.to(u.rad).value
    E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(v/2))
    rc = a * (1 - e * np.cos(E))
    ox = rc * np.cos(v)
    oy = rc * np.sin(v)
    o1x = np.sqrt(mu * a) * (-np.sin(E)) / rc
    o1y = np.sqrt(mu * a) * (np.sqrt(1 - e**2) * np.cos(E)) / rc
    X = ox*(np.cos(ω)*np.cos(Ω) - np.sin(ω)*np.cos(i)*np.sin(Ω)) - oy*(np.sin(ω)*np.cos(Ω) + np.cos(ω)*np.cos(i)*np.sin(Ω))
    Y = ox*(np.cos(ω)*np.sin(Ω) + np.sin(ω)*np.cos(i)*np.cos(Ω)) + oy*(np.cos(ω)*np.cos(i)*np.cos(Ω) - np.sin(ω)*np.sin(Ω))
    Z = ox*(np.sin(ω)*np.sin(i)) + oy*(np.cos(ω)*np.sin(i))
    Vx = o1x*(np.cos(ω)*np.cos(Ω) - np.sin(ω)*np.cos(i)*np.sin(Ω)) - o1y*(np.sin(ω)*np.cos(Ω) + np.cos(ω)*np.cos(i)*np.sin(Ω))
    Vy = o1x*(np.cos(ω)*np.sin(Ω) + np.sin(ω)*np.cos(i)*np.cos(Ω)) + o1y*(np.cos(ω)*np.cos(i)*np.cos(Ω) - np.sin(ω)*np.sin(Ω))
    Vz = o1x*(np.sin(ω)*np.sin(i)) + o1y*(np.cos(ω)*np.sin(i))
    return np.array([X, Y, Z]), np.array([Vx, Vy, Vz])
"""
The orbitcalc function converts orbital elements (semimajor axis, eccentricity, inclination, true anomaly, argument of periapsis, longitude of ascending node) 
into Cartesian position and velocity vectors in a heliocentric frame. 
It does this by first calculating the position and velocity in the orbital plane using the standard formulas, 
and then applying a rotation to account for the inclination and orientation of the orbit.
The process is outlined in more detail in the paper, but the function converts the orbital elements into position and velocity, 
each of shape (3,), with appropriate units, and returns them as a tuple of numpy arrays.
"""
def parametercalc(P,V,mu=G*c.M_sun.value):
    h = np.cross(P,V)
    h_norm = np.linalg.norm(h)
    r = np.linalg.norm(P)
    vel = np.linalg.norm(V)
    e_vec = (np.cross(V,h)/mu)-(P/r)
    e = np.linalg.norm(e_vec)
    ε = 0.5*vel**2-mu/r
    cosv = np.dot(e_vec,P)/(e*r)
    cosv = np.clip(cosv,-1,1)
    v = np.arccos(cosv)
    if np.dot(P,V) < 0:
        v = 2*np.pi - v
    v = v * (u.rad).to(u.deg)
    a = -mu / (2 * ε) * (u.m).to(u.AU)
    i = np.arccos(h[2]/h_norm) * (u.rad).to(u.deg)
    K = np.array([0,0,1])
    N = np.cross(K,h)
    N_norm = np.linalg.norm(N)
    if N_norm != 0:
        cosΩ = N[0]/N_norm
        cosΩ = np.clip(cosΩ,-1,1)
        Ω = np.arccos(cosΩ)
        if N[1] < 0:
            Ω = 2*np.pi-Ω
        Ω = Ω * (u.rad).to(u.deg)
    else:
        Ω = 0.0
    if N_norm!= 0 and e!= 0:
        cosω = np.dot(N,e_vec)/(N_norm*e)
        cosω = np.clip(cosω,-1,1)
        ω = np.arccos(cosω)
        if e_vec[2] < 0:
            ω = 2*np.pi - ω
        ω = ω * (u.rad).to(u.deg)
    else:
        ω = 0.0
    return a, e, i, v, ω, Ω
"""
The parametercalc function takes position and velocity, both shape (3,) with appropriate units,
 and an optional gravitational parameter mu (defaulting to G times the solar mass). 
 It calculates the Cartesian position and velocity vectors in a heliocentric frame 
 using the standard formulas for converting orbital elements to Cartesian coordinates. 
 The function returns a tuple of the orbital elements (semimajor axis, eccentricity, inclination, true anomaly, argument of periapsis, longitude of ascending node), each of which are floats.
"""
#Integration Methods, in case of substitution:
def rk4(p,v,m,step):
    k1p = v
    k1v = grav(p,m)
    
    k2p = v + k1v*step/2
    k2v= grav(p + k1p*step/2,m)
    
    k3p = v + k2v*step/2
    k3v= grav((p+k2p*step/2),m)
    
    k4p = v + k3v*step
    k4v = grav(p + k3p*step,m)
    
    p = p + (step/6)*(k1p+2*k2p+2*k3p+k4p)
    v = v + (step/6)*(k1v+2*k2v+2*k3v+k4v)
    return p, v
"""
The rk4 function implements the classical fourth-order Runge-Kutta integration method 
for updating the positions and velocities of bodies under gravitational acceleration. 
This was useful in testing and selection of the final integration scheme, 
but is not used in the final simulation due to its higher computational cost compared to leapfrog."""

def leapfrog(p,v,m,step):
    a = grav(p,m)
    v = v + a * step/2
    p = p + v * step
    a = grav(p,m)
    v = v + a*step/2
    return p,v
"""
Leapfrog integration function: Takes postion, velocity, mass, a timestep and computes the updated position after one timestep using the leapfrog method.
 This method is symplectic and time-reversible, making it well-suited for long-term simulations of gravitational systems. 
 Position is updated using the velocity at the half-step, and velocity is updated using the acceleration at the full step.
"""

def euler(p,v,m,step):
    a = grav(p,m)
    v = v + a*step
    p = p + v*step
    return p,v
"""
Like RK4, the Euler method was implemented for testing and comparison purposes, 
but is not used in the final simulation due to its lower accuracy and stability compared to leapfrog.
It is the simplest of integration methods, where the velocity is updated based on the current acceleration, 
and then the position is updated based on the new velocity.
"""

def timestepadapt(p,v,m,step,func):
    eref = elist[-1] 
    ptest,vtest = func(p,v,m,step)
    etest = energy(ptest,vtest,m)/estart  
    while mintol*eref<etest and etest<maxtol*eref:
        step = (1*stepadj)*step
        ptest = p
        vtest = v
        ptest,vtest = func(ptest,vtest,m,step)
        etest = energy(ptest,vtest,m)/estart  
        if etest>maxtol*eref or etest<mintol*eref:
            break   
    while etest>maxtol*eref or etest<mintol*eref:
            step = step*(1/stepadj) 
            ptest = p
            vtest = v
            ptest,vtest = func(ptest,vtest,m,step)
            etest = energy(ptest,vtest,m)/estart   
    return step    
"""
timestepadapt implements an adaptive timestep mechanism to maintain energy conservation within a specified tolerance.
The function takes the current positions, velocities, masses, initial timestep, and the integration function as inputs.
It calculates the energy of the system after a test step and compares it to the reference energy 
(most recent measurement, which in this case will just be the initial energy), 
adjusting the timestep up or down by a factor of stepadj until the energy is within the specified tolerance range (between mintol and maxtol times the reference energy).
It takes arguments p, v, m, step, func, which all have the same structure as other functions, and returns the adjusted timestep to be used for the next integration step.
"""
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Inputs:
np.random.seed(42)

#Schemes: leapfrog, euler, RK4. Initializing loop - independent variables
integrationscheme = leapfrog

startdatetime = Time.now()

stepadj = 1.1
simtimeyears = 101 #years, just over the total integration time of 100 years, to ensure we get the final parameters at 100 years.
#Time should be kept track of if the code is modified - use a set start date. Bringing this outside of the loop allows for a stable start time but this can change if something is edited.

mu = G * c.M_sun.value
    
net_time = simtimeyears*31556952

maxtol = 1+energytolerance
mintol= 1-energytolerance
    

parameterlist = np.array([[0,0,0,0],
                          [600, 0.5, 30, 10],
                          [700, 0.6, 30,10],
                          [300, 0.2, 21, 8.4],
                          [520, 0.25, 11, 4.9]])
#semimaj, eccentricity, inclination, mass (earth masses)



def kepler(E, e, M):
        return E - e*np.sin(E) - M
for simnum in range(0,50): #Beginning of loop. Data does not have to be collected in one loop like this, but it was more convient to do so, leaving the computer on overnight for a few days to collect the initial data.
    
    filename = str("Sim(" +str(simnum//10)+ ")(" + str(10+simnum%10) + ")OrbitalElements.npy") #Naming files according to the parameters used, for ease of later analysis. 
    #The first number corresponds to the row of parameterlist, and the second number corresponds to the mass of planet 9 in earth masses (10,20,...,100).

    #P9 Parameters
    semimaj = parameterlist[simnum//10,0]*u.AU
    eccentricity = parameterlist[simnum//10,1]  
    inclination = parameterlist[simnum//10,2]*u.deg
    pxmass = parameterlist[simnum//10,3]#earth masses
    trueanomaly = (180 + 18*(simnum%10))*u.deg
    periapsis = 150*u.deg
    longascending= 113*u.deg
    energytolerance = 10**-8

    #testparticle TNOs:

    #assignment and conversion of parameters for planet 9, as well as some parameters for the adaptive timestep mechanism.
    
    kbosemimajdist = np.random.uniform(150,550,3200)
    kboperi = np.random.uniform(30,50,3200)
    kboecc = 1-(kboperi/kbosemimajdist)
    kbosemimajdist=np.append(kbosemimajdist,[41,36,74])
    kboecc=np.append(kboecc,[0.5,0.3,0.9])
    inc = halfnorm.rvs(scale=15, size = 3200)
    inc=np.append(inc,[103,110,144])
    rand = np.random.uniform(0, 360, 9609)
    """
    randomization of the initial conditions for the 3200 test particles, based on the observed distribution of TNOs, 
    as well as some randomization of the angles for planet 9. Follows the process outlined in the paper.
    semimaj,eccentricity, inclination, trueanomaly, periapsis, longascending (order listed for future calls and returns and copypaste convenience)
    """
    i=0
    for i in range(len(kbosemimajdist)):
        """
        Calculates the initial position and velocity vectors for each of the 3200 test particles using the orbitcalc function, which converts from orbital elements to Cartesian coordinates.
        The true anomaly is calculated from the mean anomaly using the Kepler equation, which is solved using the Newton-Raphson method implemented in scipy's newton function. 
        The resulting position and velocity vectors are stored in testp and testv arrays, which are then combined into the initial conditions for the simulation.
        This can be done outside of the loop, but I found it takes minimal time to just rerun the simulation under a random set seed so that I didn't have to create more global variables
        """
        M = np.deg2rad(rand[3*i])
        eccanom = newton(kepler, x0=M, args=(kboecc[i], M))
        trueanom = 2*np.arctan(np.sqrt((1-kboecc[i])/(1+kboecc[i])) * np.tan(eccanom/2))
        trueanom = np.rad2deg(trueanom)
        pos, vel = orbitcalc(kbosemimajdist[i]*u.AU, kboecc[i], inc[i]*u.deg, trueanom*u.deg, rand[3*i+1]*u.deg, rand[3*i+2]*u.deg)
        if i==0:
            testp = np.array([pos])
            testv = np.array([vel])
        else:    
            testp = np.vstack((testp,pos))
            testv = np.vstack((testv,vel))
        i +=1

    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    Initialization of the positions, velocities, and masses for all bodies in the simulation, including the Sun, the 8 planets, and planet 9.
    The positions and velocities of the Sun and the 8 planets are obtained from the JPL Horizons system using astropy's get_body_barycentric_posvel function, 
    which provides accurate initial conditions for the simulation.
    The position and velocity of planet 9 are calculated using the orbitcalc function based on the specified orbital elements. 
    All of these are combined into arrays p, v, and m, which are then used as the initial conditions for the integration.
    """
    me = 5.972*10**24
    px , vx = orbitcalc(semimaj, eccentricity, inclination, trueanomaly, periapsis, longascending)
    p10 = px
    v10 = vx
    m10 = pxmass * me

    p9,v9 = get_body_barycentric_posvel("neptune", startdatetime)
    p9 = p9.xyz.to(u.m).value
    v9 = v9.xyz.to(u.m/u.s).value
    m9 = 17.15 * me

    p8,v8 = get_body_barycentric_posvel("uranus", startdatetime)
    p8 = p8.xyz.to(u.m).value
    v8 = v8.xyz.to(u.m/u.s).value
    m8 = 14.54 * me

    p7,v7 = get_body_barycentric_posvel("saturn", startdatetime)
    p7 = p7.xyz.to(u.m).value
    v7 = v7.xyz.to(u.m/u.s).value
    m7 = 95.16 * me

    p6,v6 = get_body_barycentric_posvel("Jupiter", startdatetime)
    p6 = p6.xyz.to(u.m).value
    v6 = v6.xyz.to(u.m/u.s).value
    m6 = 317.83 * me


    p5,v5 = get_body_barycentric_posvel("mars", startdatetime)
    p5 = p5.xyz.to(u.m).value
    v5 = v5.xyz.to(u.m/u.s).value
    m5 = 0.10744616724 * me

    p4,v4= get_body_barycentric_posvel("earth", startdatetime)
    p4 = p4.xyz.to(u.m).value
    v4 = v4.xyz.to(u.m/u.s).value
    m4 = me

    p3, v3 = get_body_barycentric_posvel("venus", startdatetime)
    p3=p3.xyz.to(u.m).value
    v3 = v3.xyz.to(u.m/u.s).value
    m3=0.81377046984 * me

    p2,v2 = get_body_barycentric_posvel("mercury", startdatetime)
    p2 = p2.xyz.to(u.m).value
    v2 = v2.xyz.to(u.m/u.s).value
    m2= 0.0552727638 * me

    p1,v1 = get_body_barycentric_posvel("sun", startdatetime)
    p1=p1.xyz.to(u.m).value
    v1 = v1.xyz.to(u.m/u.s).value
    m1 = 332900 * me

    m = np.array((m1,m2,m3,m4,m5,m6,m7,m8,m9,m10))
    p = np.vstack((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,testp))
    v = np.vstack((v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,testv))
    #Stack arrays into a 2d array for positions and velocities, and a 1d array for masses, 
    #to be used as initial conditions for the integration and simplify processes. Will be split later for analysis
    
    net_time = simtimeyears*31556952
    
    colorlist = ['Yellow','Red','Blue', 'Green', 'Orange', 'Purple', 'Yellow', 'Black', 'Gold', 'Plum']
    namelist = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Planet 9"]
    #Totally unnecessary for the research, but quite helpful for keeping track of the simulation and making sure it is running correctly, as well as for visualizations and sanity checks.
    step = 0.001* 31556952
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------
    
    t=0
    plist = [p.copy()]
    semi1, ecc1, inc1,_,_,_= parametercalc(p2-p1,v2-v1)
    semi2, ecc2, inc2,_,_,_= parametercalc(p3-p1,v3-v1)
    semi3, ecc3, inc3,_,_,_= parametercalc(p4-p1,v4-v1)
    semi4, ecc4, inc4,_,_,_= parametercalc(p5-p1,v5-v1)
    semi5, ecc5, inc5,_,_,_= parametercalc(p6-p1,v6-v1)
    semi6, ecc6, inc6,_,_,_= parametercalc(p7-p1,v7-v1)
    semi7, ecc7, inc7,_,_,_= parametercalc(p8-p1,v8-v1)
    semi8, ecc8, inc8,_,_,_= parametercalc(p9-p1,v9-v1)
    semi9, ecc9, inc9,_,_,_= parametercalc(p10-p1,v10-v1)
    """Calculation of initial parameters for the 9 massive bodies, to be used as the first entry in the array of orbital elements that will be saved and analyzed later.
    Angular elements are ignored."""
    
    kbosemimajdist =np.append(kbosemimajdist, np.array([semi1,semi2,semi3,semi4,semi5,semi6,semi7,semi8,semi9]))
    kboecc = np.append(np.array([ecc1,ecc2,ecc3,ecc4,ecc5,ecc6,ecc7,ecc8,ecc9]),kboecc)
    inc = np.append(np.array([inc1,inc2,inc3,inc4,inc5,inc6,inc7,inc8,inc9]),inc)
    
    """
    I append the parameters for the 9 massive bodies to the beginning of the arrays for the test particles
    so that I can save all of the parameters in one array and not have to worry about splitting them later.
    The names of the variables are a bit misleading, but it was easier to just append them to the end of the arrays I had already created for the test particles, 
    and then I can just split them later when I want to analyze the parameters for the massive bodies separately.
    """
    para = np.vstack((kbosemimajdist, kboecc, inc))
    #created a giant parameter array to be saved in one file and analyzed later, with shape (3, 3212, X) = (parameters, particles, time steps)
    tlist=[0]
    elist = [1]
    i=0
    estart = energy(p,v,m)
    step = timestepadapt(p,v,m,step,integrationscheme)
    print("Step Size=" + str(step) + " seconds, " + str((net_time/step)) + " steps expected") 
    passed1 = passed10 = passed30 = passed100 = False
    listnum=1
    printing = False
    while t < net_time: 
        p,v = integrationscheme(p,v,m,step)
        if (i%200==0) or (passed1==False and t > 31556952)or (passed10 == False and t > 10*31556952) or (passed30 == False and t > 30*31556952) or (passed100 == False and t > 100*31556952):
            if t>100*31556952 and passed100==False:
                passed100 = True
                tlist.append(listnum)
                printing = True
            elif t>30*31556952 and passed30==False:
                passed30 = True
                tlist.append(listnum)
                printing = True
            elif t>10*31556952 and passed10==False:
                passed10 = True
                tlist.append(listnum)
                printing = True
            elif t>31556952 and passed1==False:
                passed1 = True
                tlist.append(listnum)
                printing = True
            
            """
            This is just a way to save the parameters at specific time intervals (1 year, 10 years, 30 years, 100 years) 
            as well as every 200 steps, to ensure that I have enough data points to analyze the evolution of the system over time,
            Printing the location in the final parameter array where the parameters at these specific time intervals are saved, 
            ensuring that I can accuately analyze the parameters at these time intervals later
            I also used this to ensure that the simulation was running correctly, as it let me know as simulations were being run and if energy was being conserved.
            """
            
            listnum += 1
            arrsemimaj = np.zeros(p.shape[0] - 1)
            arreccentricity = np.zeros(p.shape[0] - 1)
            arrinclination = np.zeros(p.shape[0] - 1)
            
            for jj in range (1,p.shape[0]):
                semimaj, eccentricity, inclination, _,_,_= parametercalc(p[jj,:]-p[0,:],v[jj,:]-v[0,:])
                arreccentricity[jj-1] = eccentricity
                arrinclination[jj-1] = inclination
                arrsemimaj[jj-1] = semimaj
            
            para1step = np.vstack((arrsemimaj,arreccentricity,arrinclination))
            para = np.dstack((para,para1step))
            
            #Collection of new parameters and adding to the data
            if printing ==True:
                print(str(i) + " Steps Complete: " + str(t*100/net_time) + "% of Time Completed. Energy Accuracy: " + str(100*energy(p,v,m)/estart)+ "% of Original")
            printing = False
        i += 1    
        t += step

    arrsemimaj      = np.zeros(p.shape[0] - 1)
    arreccentricity = np.zeros(p.shape[0] - 1)
    arrinclination  = np.zeros(p.shape[0] - 1)
    
    for jj in range(p.shape[0] - 1): 
        semimaj, eccentricity, inclination, _,_,_ = parametercalc(p[jj+1,:] - p[0,:], v[jj+1,:] - v[0,:])
        arrsemimaj[jj]= semimaj
        arreccentricity[jj] = eccentricity
        arrinclination[jj]  = inclination
        
    para1step = np.vstack((arrsemimaj, arreccentricity, arrinclination))
    para = np.dstack((para, para1step))
    #Final data collection.
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    
    print("Sim Number " + str(simnum) + " complete. Total Steps:" + str(i) + ", Final Energy Accuracy:" + str(100*energy(p,v,m)/estart) + "% of Original")
    print(tlist)
    np.save(filename, para) 
    #shape is (3, 3212, X) = (parameters, particles, time steps)

endtime = time.time()

print("Total Runtime: " + str(endtime - starttime) + " seconds") 
#Just to make sure I know how long the simulations are taking, as they took a very long time.

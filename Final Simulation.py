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

def energy(p, v, m):
    nummassive = len(m)

    ptemp = p[:nummassive]
    vtemp = v[:nummassive]

    vsquarebyobj = np.sum(vtemp**2, axis=1)
    KE = np.sum(vsquarebyobj * m * 0.5)

    p_rel = ptemp[:, None, :] - ptemp[None, :, :]
    dist = np.linalg.norm(p_rel, axis=2)
    np.fill_diagonal(dist, np.inf)

    MxM = m[:, None] * m[None, :]

    U_pairwise = -G * MxM / dist
    PE = np.sum(U_pairwise * 0.5)
    return PE + KE

"""
energy(p,v,m) calculates the total energy of the system given the positions p (shape (N,3)), velocities v (shape (N,3)), and masses m (shape (N,)) of the bodies.
The function first separates the massive bodies from the test particles and calculates the kinetic energy (KE) of the massive bodies using their velocities and masses.
Then it calculates the pairwise potential energy (PE) between the massive bodies by computing their relative positions and distances
, and using the gravitational potential energy formula. The function sums up the kinetic and potential energy to return the total energy of the system.
"""

def xrotate(array, theta):
    theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[1,0,0],
    [0,c,-s],
    [0,s,c]])
    return array @ R

def orbitcalc(semimaj, eccentricity, inclination, longascending, periapsis, trueanomaly, mu=G*c.M_sun.value):
    # Convert angles to radians
    a = semimaj * u.au.to(u.m)
    e = eccentricity
    i = np.deg2rad(inclination)
    Ω = np.deg2rad(longascending)
    ω = np.deg2rad(periapsis)
    v = np.deg2rad(trueanomaly)

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
    v = np.rad2deg(v)

    a = -mu / (2 * ε) * (u.m).to(u.AU)
    
    cosi = h[2] / h_norm
    cosi = np.clip(cosi, -1, 1)
    i = np.rad2deg(np.arccos(cosi))

    K = np.array([0,0,1])
    N = np.cross(K,h)
    N_norm = np.linalg.norm(N)

    if N_norm != 0:
        cosΩ = N[0]/N_norm
        cosΩ = np.clip(cosΩ,-1,1)
        Ω = np.arccos(cosΩ)
       
        if N[1] < 0:
            Ω = 2*np.pi-Ω
        Ω = np.rad2deg(Ω)
    else:
        Ω = 0.0 #Conditional was put in due to errors in test cases, although it probably could be fixed in other ways
       
    if N_norm!= 0 and e!= 0: #This could be nested within the first condition but it is easier to visually interpret this way
        cosω = np.dot(N,e_vec)/(N_norm*e)
        cosω = np.clip(cosω,-1,1)
        ω = np.arccos(cosω)
       
    cosomega = np.dot(N, e_vec) / (N_norm * e)
    sinomega = np.dot(np.cross(N, e_vec), h) / (N_norm * e * h_norm)
    omega = np.arctan2(sinomega, cosomega)

    if omega < 0:
        omega += 2*np.pi

    ω = np.rad2deg(omega)
       
    return a, e, i, Ω, ω, v

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
but is not used in the final simulation due to its higher computational cost compared to leapfrog.
"""

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

    ptest,vtest = func(p,v,m,step)
    etest = energy(ptest,vtest,m)/estart
       
    while etest>maxtol or etest<mintol:
            step /= stepadj
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
adjusting the timestep by a factor of stepadj until the energy is within the specified tolerance range (between mintol and maxtol times the reference energy).
It takes arguments p, v, m, step, func, which all have the same structure as other functions, and returns the adjusted timestep to be used for the next integration step.
"""

#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Inputs:
np.random.seed(42)

#Schemes: leapfrog, euler, RK4. Initializing loop - independent variables
integrationscheme = leapfrog

startdatetime = Time("2026-09-23 00:05:00", format="iso", scale="utc")

stepadj = 1.1
simtimeyears = 101 #years, just over the total integration time of 100 years, to ensure we get the final parameters at 100 years.
#Time should be kept track of if the code is modified - use a set start date. Bringing this outside of the loop allows for a stable start time but this can change if something is edited.

mu = G * c.M_sun.value

net_time = simtimeyears*31556952
energytolerance = 10**-8
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
print("REACHED SIMULATION LOOP")
for simnum in range(20,100): #Beginning of loop. Data does not have to be collected in one loop like this, but it was more convient to do so, leaving the computer on overnight for a few days to collect the initial data.
    np.random.seed(42)
    filename = str(f"Sim({simnum//20})({simnum%20})OrbitalElements.npy") #Naming files according to the parameters used, for ease of later analysis.
    #The first number corresponds to the row of parameterlist, and the second number corresponds to the mass of planet 9 in earth masses (10,20,...,100).
    file2 = str(f"Sim({simnum//20})({simnum%20})FinalPositions.npy")
    #P9 Parameters
    semimaj = parameterlist[simnum//20,0]

    eccentricity = parameterlist[simnum//20,1]

    inclination = parameterlist[simnum//20,2]

    pxmass = parameterlist[simnum//20,3]#earth masses

    trueanomaly = (18*(simnum%20))

    periapsis = 150

    longascending= 113


        #-------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    Initialization of the positions, velocities, and masses for all bodies in the simulation, including the Sun, the 8 planets, and planet 9.
    The positions and velocities of the Sun and the 8 planets are obtained from the JPL Horizons system using astropy's get_body_barycentric_posvel function,
    which provides accurate initial conditions for the simulation.
    The position and velocity of planet 9 are calculated using the orbitcalc function based on the specified orbital elements.
    All of these are combined into arrays p, v, and m, which are then used as the initial conditions for the integration.
    """
    me = 5.972*10**24 #kg

    p10 , v10 = orbitcalc(semimaj, eccentricity, inclination, longascending, periapsis,trueanomaly)
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

    p1,v1 = get_body_barycentric_posvel("sun", startdatetime)
    p1=p1.xyz.to(u.m).value
    v1 = v1.xyz.to(u.m/u.s).value
    m1 = 332900 * me

    #Convert to ecliptic coordinates:
    p9 = xrotate(p9,23.4297)
    v9 = xrotate(v9,23.4297)

    p8 = xrotate(p8,23.4297)
    v8 = xrotate(v8,23.4297)

    p7 = xrotate(p7,23.4297)
    v7 = xrotate(v7,23.4297)

    p6 = xrotate(p6,23.4297)
    v6 = xrotate(v6,23.4297)

    p1 = xrotate(p1,23.4297)
    v1 = xrotate(v1,23.4297)

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
    nul = np.array([])
    Ol = np.array([])
    ol = np.array([])

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

        Ol=np.append(Ol,rand[3*i+2])
        ol=np.append(ol,rand[3*i+1])  
        nul=np.append(nul,trueanom)
       
        pos, vel = orbitcalc(kbosemimajdist[i], kboecc[i], inc[i], rand[3*i+2], rand[3*i+1], trueanom)
       
        if i==0:
            testp = np.array([pos+p1])
            testv = np.array([vel+v1])
        else:
            testp = np.vstack((testp,pos+p1))
            testv = np.vstack((testv,vel+v1))
           
        i +=1

    m = np.array([m1,m6,m7,m8,m9,m10])
    p = np.vstack((p1,p6,p7,p8,p9,p10,testp))
    v = np.vstack((v1,v6,v7,v8,v9,v10,testv))

    #Stack arrays into a 2d array for positions and velocities, and a 1d array for masses,
    #to be used as initial conditions for the integration and simplify processes. Will be split later for analysis

    net_time = simtimeyears*31556952

    colorlist = ['Orange', 'Purple', 'Yellow', 'Black', 'Gold', 'Plum']
    namelist = ["Jupiter", "Saturn", "Uranus", "Neptune", "Planet 9"]
    step = 0.001* 31556952

    #------------------------------------------------------------------------------------------------------------------------------------------------------

    t=0
    plist = [p.copy()]
    tlist = [0]
    semi5, ecc5, inc5,O5,o5,nu5= parametercalc(p6-p1,v6-v1)
    semi6, ecc6, inc6,O6,o6,nu6= parametercalc(p7-p1,v7-v1)
    semi7, ecc7, inc7,O7,o7,nu7= parametercalc(p8-p1,v8-v1)
    semi8, ecc8, inc8,O8,o8,nu8= parametercalc(p9-p1,v9-v1)
    semi9, ecc9, inc9,O9,o9,nu9= parametercalc(p10-p1,v10-v1)
    """Calculation of initial parameters for the 9 massive bodies, to be used as the first entry in the array of orbital elements that will be saved and analyzed later.
    Angular elements are ignored."""

    semimaj =np.append(np.array([semi5,semi6,semi7,semi8,semi9]),kbosemimajdist)
    ecc = np.append(np.array([ecc5,ecc6,ecc7,ecc8,ecc9]),kboecc)
    inc = np.append(np.array([inc5,inc6,inc7,inc8,inc9]),inc)
    Omega = np.append(np.array([O5,O6,O7,O8,O9]),Ol)
    omega = np.append(np.array([o5,o6,o7,o8,o9]),ol)
    nu = np.append(np.array([nu5,nu6,nu7,nu8,nu9]),nul)

    """
    I append the parameters for the 9 massive bodies to the beginning of the arrays for the test particles
    so that I can save all of the parameters in one array and not have to worry about splitting them later.
    The names of the variables are a bit misleading, but it was easier to just append them to the end of the arrays I had already created for the test particles,
    and then I can just split them later when I want to analyze the parameters for the massive bodies separately.
    """
    para = np.vstack((semimaj, ecc, inc, Omega, omega, nu))
    #created a giant parameter array to be saved in one file and analyzed later, with shape (3, 3212, X) = (parameters, particles, time steps)
    tlist=[0]
    i=0
    estart = energy(p,v,m)
    elist = [1]
    print(estart)
    step = 0.625 * 60*60*24 #timestepadapt(p,v,m,step,integrationscheme)
    print("Step Size=" + str(step) + " seconds, " + str((net_time/step)) + " steps expected")

    passed1 = passed10 = passed30 = passed100 = False
    listnum=1
    printing = False

    print("REACHED SIMULATION LOOP")
    while t < net_time:
        p,v = integrationscheme(p,v,m,step)
       
        if (passed1==False and t > 31556952)or (passed10 == False and t > 10*31556952) or (passed30 == False and t > 30*31556952) or (passed100 == False and t > 100*31556952):
           
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
            arrO = np.zeros(p.shape[0] - 1)
            arro = np.zeros(p.shape[0] - 1)
            arrnu = np.zeros(p.shape[0] - 1)
           
            for jj in range (1,p.shape[0]):
                semimaj, eccentricity, inclination, O,o,nu= parametercalc(p[jj,:]-p[0,:],v[jj,:]-v[0,:])
                arreccentricity[jj-1] = eccentricity
                arrinclination[jj-1] = inclination
                arrsemimaj[jj-1] = semimaj
                arrO[jj-1]=O
                arro[jj-1]=o
                arrnu[jj-1]=nu
               
            para1step = np.vstack((arrsemimaj,arreccentricity,arrinclination, arrO, arro, arrnu))
            para = np.dstack((para,para1step))
           
            #Collection of new parameters and adding to the data
           
            if printing ==True:
                print(str(i) + " Steps Complete: " + str(t*100/net_time) + "% of Time Completed. Energy Accuracy: " + str(100*energy(p,v,m)/estart)+ "% of Original")
            printing = False
           
        i += 1    
        t += step

    arrsemimaj = np.zeros(p.shape[0] - 1)
    arreccentricity = np.zeros(p.shape[0] - 1)
    arrinclination = np.zeros(p.shape[0] - 1)
    arrO = np.zeros(p.shape[0] - 1)
    arro = np.zeros(p.shape[0] - 1)
    arrnu = np.zeros(p.shape[0] - 1)

    for jj in range (1,p.shape[0]):
            semimaj, eccentricity, inclination,O,o,nu= parametercalc(p[jj,:]-p[0,:],v[jj,:]-v[0,:])
            arreccentricity[jj-1] = eccentricity
            arrinclination[jj-1] = inclination
            arrsemimaj[jj-1] = semimaj
            arrO[jj-1]=O
            arro[jj-1]=o
            arrnu[jj-1]=nu
           
    para1step = np.vstack((arrsemimaj,arreccentricity,arrinclination, arrO, arro, arrnu))
    para = np.dstack((para,para1step))

    #Final data collection.

#-------------------------------------------------------------------------------------------------------------------------------------------------------

    print("Sim Number " + str(simnum) + " complete. Total Steps:" + str(i) + ", Final Energy Accuracy:" + str(100*energy(p,v,m)/estart) + "% of Original")
    print(tlist)
    np.save(filename, para)
    np.save(file2, p)
    #shape is (6, 3209, X) = (parameters, particles, time steps)


endtime = time.time()

print("Total Runtime: " + str(endtime - starttime) + " seconds")
#Just to make sure I know how long the simulations are taking, as they took a very long time

'''
import scipy.stats as stats
parray = np.zeros((4, 20,6))

#shape is (6, 3212, X) = (parameters, particles, time steps)
Opara = np.load('Sim(0)(19)OrbitalElements.npy')
for m in range (4):
     for n in range (20):
          para = np.load(f'Sim({m+1})({n})OrbitalElements.npy')
          for i in range(6):
              O = Opara[i, 4:, -1]
              P = para[i, 5:, -1]
              p = stats.ks_2samp(O, P)
              parray[m,n,i] = p.pvalue
             
print(parray)
fig = plt.figure(figsize=(12, 5))
plt.scatter(histecc, histinc)
plt.xlabel('Eccentricity KS p-value')
plt.ylabel('Inclination KS p-value')
plt.title('KS Test p-values for Final Eccentricity and Inclination Distributions')
plt.show()
'''

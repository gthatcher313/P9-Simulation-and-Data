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
#Core Functions 4 this
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

def leapfrog(p,v,m,step):
    a = grav(p,m)
    v = v + a * step/2
    p = p + v * step
    a = grav(p,m)
    v = v + a*step/2
    return p,v

def euler(p,v,m,step):
    a = grav(p,m)
    v = v + a*step
    p = p + v*step
    return p,v

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
#-------------------------------------------------------------------------------------------------------------------------------------------------------
#Inputs:
np.random.seed(42)

#Schemes: leapfrog, euler, RK4
integrationscheme = leapfrog

parameterlist = np.array([[600, 0.5, 30, 0],
                          [600, 0.5, 30, 10],
                          [700, 0.6, 30,10],
                          [300, 0.2, 21, 8.4],
                          [520, 0.25, 11, 4.9]])
#semimaj, eccentricity, inclination, mass (earth masses)

def kepler(E, e, M):
        return E - e*np.sin(E) - M
for simnum in range(10*parameterlist.shape[0]):
    #P9 parameters:
    filename = str("Sim(" +str(simnum//10)+ ")(" + str(10+simnum%10) + ")OrbitalElements.npy")
    if simnum==0:
        print(filename + " = Example Name")
    semimaj = parameterlist[simnum//10,0]*u.AU
    eccentricity = parameterlist[simnum//10,1]  
    inclination = parameterlist[simnum//10,2]*u.deg
    pxmass = parameterlist[simnum//10,3]#earth masses
    trueanomaly = (180 + 18*(simnum%10))*u.deg
    periapsis = 150*u.deg
    longascending= 113*u.deg
    energytolerance = 10**-8
    if simnum==0:
        energytolerance = 10**-9
    stepadj = 1.1 

    startdatetime = Time.now()
    simtimeyears = 101 #years
    #testparticle TNOs:

    i=0
    kbosemimajdist = np.random.uniform(150,550,3200)
    kboperi = np.random.uniform(30,50,3200)
    kboecc = 1-(kboperi/kbosemimajdist)
    kbosemimajdist=np.append(kbosemimajdist,[41,36,74])
    kboecc=np.append(kboecc,[0.5,0.3,0.9])
    inc = halfnorm.rvs(scale=15, size = 3200)
    inc=np.append(inc,[103,110,144])
    rand = np.random.uniform(0, 360, 9609)


    #semimaj,eccentricity, inclination, trueanomaly, periapsis, longascending for future calls and returns

    for i in range(len(kbosemimajdist)):
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
    mu = G * c.M_sun.value
    names = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"]

    net_time = simtimeyears*31556952

    maxtol = 1+energytolerance
    mintol= 1-energytolerance
    colorlist = ['Yellow','Red','Blue', 'Green', 'Orange', 'Purple', 'Yellow', 'Black', 'Gold', 'Plum']
    namelist = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Planet 9"]
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
    kbosemimajdist =np.append(kbosemimajdist, np.array([semi1,semi2,semi3,semi4,semi5,semi6,semi7,semi8,semi9]))
    kboecc = np.append(np.array([ecc1,ecc2,ecc3,ecc4,ecc5,ecc6,ecc7,ecc8,ecc9]),kboecc)
    inc = np.append(np.array([inc1,inc2,inc3,inc4,inc5,inc6,inc7,inc8,inc9]),inc)

    para = np.vstack((kbosemimajdist, kboecc, inc))
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
    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    print("Sim Number " + str(simnum) + " complete. Total Steps:" + str(i) + ", Final Energy Accuracy:" + str(100*energy(p,v,m)/estart) + "% of Original")
    print(tlist)
    np.save(filename, para) 
    # shape is (3, 3212, X) = (parameters, particles, time steps)


# Opara = np.load('Sim0OrbitalElements.npy')

# para = para - Opara[:,:,0][:,:,None]

# fig, axes = plt.subplots(3, 2, figsize=(30,20))
# ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

# timeecc = np.zeros(para.shape[2])
# timeincl = np.zeros(para.shape[2])
# eccstd = np.zeros(para.shape[2])
# inclstd = np.zeros(para.shape[2])
# kk=0
# for k in range(para.shape[2]):
#     timeecc[kk] = np.mean(para[1,:,k])
#     timeincl[kk]= np.mean(para[2,:,k])
#     eccstd[kk] = np.std(para[1,:,k])
#     inclstd[kk]= np.std(para[2,:,k])
#     kk+=1

# ax1.scatter(np.linspace(0,simtimeyears,para.shape[2]),timeecc)
# ax1.errorbar(np.linspace(0,simtimeyears,para.shape[2]), timeecc, yerr=eccstd, fmt='o', markersize=2, capsize=2)
# ax2.scatter(np.linspace(0,simtimeyears,para.shape[2]),timeincl)
# ax2.errorbar(np.linspace(0,simtimeyears,para.shape[2]), timeincl, yerr=inclstd, fmt='o', markersize=2, capsize=2)

# ax1.set_xlabel("Time (Years)")
# ax2.set_xlabel("Time (Years)")
# ax1.set_ylabel("Eccentricity")
# ax2.set_ylabel("Inclination")


# ax3.scatter(para[1,:,tlist[1]],para[2,:,tlist[1]], s = 1)
# ax4.scatter(para[1,:,tlist[2]],para[2,:,tlist[2]], s = 1)
# ax5.scatter(para[1,:,tlist[3]],para[2,:,tlist[3]], s = 1)
# ax6.scatter(para[1,:,tlist[4]],para[2,:,tlist[4]], s = 1)

# ax3.set_xlabel("1 Year Eccentricity")
# ax3.set_ylabel("1 Year Inclination")
# ax4.set_xlabel("10 Year Eccentricity")
# ax4.set_ylabel("10 Year Inclination")
# ax5.set_xlabel("30 Year Eccentricity")
# ax5.set_ylabel("30 Year Inclination")
# ax6.set_xlabel("100 Year Eccentricity")
# ax6.set_ylabel("100 Year Inclination")


# plt.show()

# #------------------------------------------------------------------------------------------------------------------------------------------------------

endtime = time.time()
print("Total Runtime: " + str(endtime - starttime) + " seconds")
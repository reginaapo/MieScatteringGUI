# Calling Necessary packages
import importlib.resources
import numpy as np
import matplotlib.pyplot as plt
import miepython
from scipy.integrate import quad
from scipy.optimize import curve_fit
from numpy import sin
from numpy import cos
from numpy import exp
import random
import tkinter
from tkinter import *

import CTkMessagebox as tkMessageBox

import importlib.resources
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.backends
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import miepython
from scipy.integrate import quad
from scipy.optimize import curve_fit
from numpy import sin
from numpy import cos
from numpy import exp

#Importing all relevant Refractive index data
nnamesio2 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT 2023\refractiveIndexSiO2.csv"
sio2 = np.genfromtxt(nnamesio2, delimiter=',', skip_header=1)
sio2_lam = sio2[:, 0]
sio2_mre = sio2[:, 1]

nnameal2o3 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT 2023\refractiveIndexAl2O3.csv"
al2o3 = np.genfromtxt(nnameal2o3, delimiter=',', skip_header=1)
al2o3_lam = al2o3[:, 0]
al2o3_mre = al2o3[:, 1]

nnamefeo = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT 2023\refractiveIndexFeO.csv"
feo = np.genfromtxt(nnamefeo, delimiter=',', skip_header=1)
feo_lam = feo[:, 0]
feo_lam = feo_lam[:int(len(feo_lam) / 2)]
feo_mre = feo[:, 1]
feo_mre = feo_mre[:int(len(feo_mre) / 2)]

nnamemgo = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT 2023\refractiveIndexMgO.csv"
mgo = np.genfromtxt(nnamemgo, delimiter=',', skip_header=1)
mgo_lam = mgo[:, 0]
mgo_mre = mgo[:, 1]

nnamemno2 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT 2023\refractiveIndexMnO2.csv"
mno2 = np.genfromtxt(nnamemno2, delimiter=',', skip_header=1)
mno2_lam = mno2[:, 0]
mno2_mre = mno2[:, 1]

nnametio2 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT 2023\refractiveIndexTiO2.csv"
tio2 = np.genfromtxt(nnametio2, delimiter=',', skip_header=1)
tio2_lam = tio2[:, 0]
tio2_mre = tio2[:, 1]

nnamecao = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT 2023\refractiveIndexCaO.csv"
cao = np.genfromtxt(nnamecao, delimiter=',', skip_header=1)
cao_lam = cao[:, 0]
cao_mre = cao[:, 1]

nname95 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid95.txt"
h2so495 = np.genfromtxt(nname95, delimiter='\t', skip_header=1)
h2so4_lam95 = h2so495[:, 0]
h2so4_mre95 = h2so495[:, 1]

nname25 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid25.txt"
h2so425 = np.genfromtxt(nname25, delimiter='\t', skip_header=1)
h2so4_lam25 = h2so425[:,0]
h2so4_mre25 = h2so425[:,1]

nname38 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid38.txt"
h2so438 = np.genfromtxt(nname38, delimiter='\t', skip_header=1)
h2so4_lam38 = h2so438[:, 0]
h2so4_mre38 = h2so438[:, 1]

nname5 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid50.txt"
h2so45 = np.genfromtxt(nname5, delimiter='\t', skip_header=1)
h2so4_lam5 = h2so45[:,0]
h2so4_mre50 = h2so45[:,1]


nname6 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid61.txt"
h2so46 = np.genfromtxt(nname6, delimiter='\t', skip_header=1)
h2so4_lam6 = h2so46[:,0]
h2so4_mre60 = h2so46[:,1]


nname75 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid75.txt"
h2so475 = np.genfromtxt(nname75, delimiter='\t', skip_header=1)
h2so4_lam75 = h2so475[:,0]
h2so4_mre75 = h2so475[:,1]


nname84 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid84.txt"
h2so484 = np.genfromtxt(nname84, delimiter='\t', skip_header=1)
h2so4_lam84= h2so484[:,0]
h2so4_mre84 = h2so484[:,1]


nname9 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid90.txt"
h2so49 = np.genfromtxt(nname9, delimiter='\t', skip_header=1)
h2so4_lam9 = h2so49[:,0]
h2so4_mre90 = h2so49[:,1]

nname0 = r"C:\Users\regin\OneDrive - Massachusetts Institute of Technology\Documents\MIT2022\MorningStar\refractiveIndexSulfuricAcid00.txt"
h2o = np.genfromtxt(nname0, delimiter='\t', skip_header=1)
h2o_lam = h2o[:,0]
h2o_mre = h2o[:,1]

#Defining necessary function
def find_nearest(array, value):
   array = np.asarray(array)
   idx = (np.abs(array - value)).argmin()
   return array[idx]

#Creating GUI
root = tkinter.Tk()
tkinter.Tk.wm_title(root, "Mie Scattering Simulation")
frame1 = tkinter.Frame(root)
frame2 = tkinter.Frame(root)
frame3 = tkinter.Frame(root)
frame4 = tkinter.Frame(root)



# Pack the frames side by side along with their titles
frame1.pack(side=tkinter.LEFT)
T1 = Label(frame1, text="Experimental Setup: ", font=("Helvetica", 18, "bold"))
T1.pack()
frame2.pack(side=tkinter.LEFT)
T2 = Label(frame2, text="Aerosol Parameters: ", font=("Helvetica", 18, "bold"))
T2.pack( )
frame3.pack(side=tkinter.LEFT)
T3 = Label(frame3, text="Graphs: ", font=("Helvetica", 18, "bold"))
T3.pack( )
frame4.pack(side=tkinter.LEFT)

#Now we start putting things into each frame

L1S = Label(frame2, text="Select a Aerosol to simulate: ")
L1S.pack( )
def show():
   label.config(text=clickedS.get())
# Dropdown menu options
minerals=["H2O","H2SO4-25","H2SO4-38","H2SO4-50","H2SO4-75","H2SO4-84","H2SO4-95","SiO2","TiO2","Al2O3","FeO","MnO","CaO","K2O","MgO","Ph3","No Mineral"]
mMineral=[h2o_mre,h2so4_mre25,h2so4_mre38,h2so4_mre50,h2so4_mre75,h2so4_mre84,h2so4_mre95,sio2_mre,tio2_mre,al2o3_mre,feo_mre,mno2_mre,cao_mre,0,mgo_mre,[2.144],0]
lamMineral=[h2o_lam/1000,h2so4_lam25/1000,h2so4_lam38/1000,h2so4_lam5/1000,h2so4_lam75/1000,h2so4_lam84/1000,h2so4_lam95/1000,sio2_lam,tio2_lam,al2o3_lam,feo_lam,mno2_lam/1000,cao_lam/1000,0,mgo_lam,h2so4_lam95/1000,0]
# datatype of menu text
clickedS = StringVar()
# initial menu text
clickedS.set("H2SO4-95")
# Create Dropdown menu
dropS = OptionMenu(frame2, clickedS, *minerals)
dropS.pack()

kmS=minerals.index(clickedS.get())
kAerosol=mMineral[kmS]
sAerosol=DoubleVar()

L1 = Label(frame2, text="Select a mineral to Simulate: ")
L1.pack( )
def show():
   label.config(text=clickedM.get())
# Dropdown menu options

# datatype of menu text
clickedM = StringVar()
# initial menu text
clickedM.set("No Mineral")
# Create Dropdown menu
drop = OptionMenu(frame2, clickedM, *minerals)
drop.pack()
if clickedM.get() == 'No Mineral':
   km=minerals.index(clickedS.get())
else:
   km=minerals.index(clickedM.get())
kMineral=mMineral[km]
sMineral=DoubleVar()
aS = DoubleVar()
sWavelen=DoubleVar()

L2 = Label(frame2, text="Select if you want to graph a range or a specific value for the Particle Size (um): ")
L2.pack()
typeA=["Range","Value"]
clickedA = StringVar()
clickedA.set("Value")
var = DoubleVar()
scale = Scale(frame2, label="Particle Size (um):", variable=var, orient=HORIZONTAL, from_=1, to=20, resolution=0.1)
label = Label(frame2)
avarMin = DoubleVar()
avarMax = DoubleVar()
L21 = Label(frame2, text="Select minimum particle Size (um): ")
L22 = Label(frame2, text="Select maximum particle Size (um): ")
E1 = Spinbox(frame2, from_=1, to=20)
#avarMin=float(E1.get())
E2 = Spinbox(frame2, from_=1, to=20)
#avarMax=float(E2.get())


def sel():
   global aS
   aS = var.get()
def Rsel():
   global aS
   #aS = []
   avarMin = float(E1.get())
   avarMax = float(E2.get())
   aS = np.arange(avarMin, avarMax, 0.1)

button = Button(frame2, text="Apply", command=sel)
Rbutton = Button(frame2, text="Apply", command=Rsel)
def selA(clickedA):
   if (str(clickedA)=="Value"):
      Rbutton.pack_forget()
      E1.pack_forget()
      E2.pack_forget()
      L21.pack_forget()
      L22.pack_forget()
      label.pack()
      scale.pack()
      button.pack(anchor=CENTER)

   if (str(clickedA)=="Range"):
      button.pack_forget()
      label.pack_forget()
      scale.pack_forget()
      L21.pack()
      E1.pack()
      L22.pack()
      E2.pack()
      Rbutton.pack(anchor=CENTER)
dropA = OptionMenu(frame2, clickedA, *typeA, command=selA)
dropA.pack()


anvarMin = DoubleVar()
anvarMax = DoubleVar()
anMinL = Label(frame1, text="Select minimum angle (°): ")
anMaxL = Label(frame1, text="Select maximum angle (°): ")
anMin =StringVar(frame1)
anMin.set("168.0")
anMinS = Spinbox(frame1, from_=0, to=180, textvariable=anMin)
anMax =StringVar(frame1)
anMax.set("173.1")
anMaxS = Spinbox(frame1, from_=0, to=180, textvariable=anMax)

winR = DoubleVar()
focD = DoubleVar()
winRL = Label(frame1, text="Select Window Radius (mm): ")
focDL = Label(frame1, text="Select Focal Distance (mm): ")
winR0 =StringVar(frame1)
winR0.set("87.322")
winRS = Spinbox(frame1, from_=0, to=200,  increment=0.01, textvariable=winR0)
focD0 =StringVar(frame1)
focD0.set("250")
focDS = Spinbox(frame1, from_=0, to=500, textvariable=focD0)


anMinL.pack()
anMinS.pack()
anMaxL.pack()
anMaxS.pack()
winRL.pack()
winRS.pack()
focDL.pack()
focDS.pack()

L3 = Label(frame1, text="Select if you want to graph a range or a specific value for the Wavelength (nm): ")
L3.pack()
typeW=["Range","Value"]
clickedW = StringVar()
clickedW.set("Value")
wvar = DoubleVar()
wscale = Scale(frame1, label="Wavelength (nm):", variable=wvar, orient=HORIZONTAL, from_=400, to=620, resolution=0.1)
wlabel = Label(frame1)
wvarMin = DoubleVar()
wvarMax = DoubleVar()
L31 = Label(frame1, text="Select minimum wavelength (nm): ")
L32 = Label(frame1, text="Select maximum wavelength (nm): ")
E31 = Spinbox(frame1, from_=400, to=600)
E32 = Spinbox(frame1, from_=400, to=600)
#wvarMin=E31.get()
#wvarMax=E32.get()


def wsel():
   global sWavelen, sMineral, sAerosol
   newlambda = find_nearest(lamMineral[km], (wvar.get() / 1000))
   sWavelen = newlambda
   newlambdaA = find_nearest(lamMineral[kmS], (wvar.get() / 1000))
   sAerosol = newlambdaA
   [j] = np.where(lamMineral[km] == newlambda)
   [i] = np.where(lamMineral[kmS] == newlambdaA)
   [sMineral] = kMineral[j]
   [sAerosol] = kAerosol[i]
def wRsel():
   global sWavelen, sMineral, sAerosol
   wvarMin = float(E31.get())
   wvarMax = float(E32.get())
   i = wvarMin / 1000
   sMineral = []
   sWavelen = []
   sAerosol = []

   indJ = []
   while i <= wvarMax / 1000:
      newlambda = find_nearest(lamMineral[km], i)
      [j] = np.where(lamMineral[km] == newlambda)
      if newlambda in sWavelen:
         i += 1 / 1000
      else:
         sWavelen.append(newlambda)
         indJ.append(j)
         i += 1 / 1000
   for l in range(len(lamMineral[km])):
      if l in indJ:
         sMineral.append(kMineral[l])
         sAerosol.append(kAerosol[l])
wbutton= Button(frame1, text="Apply", command=wsel)
wRbutton = Button(frame1, text="Apply", command=wRsel)
def selW(clickedW):
   if (str(clickedW)=="Value"):
      wRbutton.pack_forget()
      E31.pack_forget()
      E32.pack_forget()
      L31.pack_forget()
      L32.pack_forget()
      wlabel.pack()
      wscale.pack()
      wbutton.pack(anchor=CENTER)

   if (str(clickedW)=="Range"):
      wbutton.pack_forget()
      wlabel.pack_forget()
      wscale.pack_forget()
      L31.pack()
      E31.pack()
      L32.pack()
      E32.pack()
      wRbutton.pack(anchor=CENTER)
dropW = OptionMenu(frame1, clickedW, *typeW, command=selW)
dropW.pack()



L4 = Label(frame3, text="Select the types of Graphs you want to visualize: ")
L4.pack()


def isfloat(num):
   try:
      float(num)
      return True
   except TypeError:
      return False



def allG(m_sphere0, a0, lambda0, m_aerosol0, gVec0, eVec0):
   newWindow = Toplevel(root)
   newWindow.title("Mie Scattering Simulation Results")
   newWindow.geometry("900x800")
   global figs, canvas, subplots
   figs = plt.Figure(figsize=(16, 8), dpi=100)
   cm = plt.get_cmap('RdYlBu')
   markers=["d", "v", "s", "*", "^", "o", "<", ">", "x", "+","."]
   canvas = FigureCanvasTkAgg(figs, newWindow)
   if isfloat(a0) == True:
      print('Selected Particle Size: %s' % a0)
      if isfloat(lambda0) == True:
         print('Selected Wavelength: %s' % lambda0)
      elif isfloat(lambda0) == False:
         print('Selected Wavelength: (%s,%s)' % (min(lambda0), max(lambda0)))
   elif isfloat(a0) == False:
      print('Selected Particle Size: (%s,%.1f)' % (min(a0), max(a0)))
      if isfloat(lambda0) == True:
         print('Selected Wavelength: %s' % lambda0)
      elif isfloat(lambda0) == False:
         print('Selected Wavelength: (%s,%s)' % (min(lambda0), max(lambda0)))
   # theta = np.linspace(-180,180,1800)
   theta = np.arange(-181, 181, 1)
   numG = gVec0.count(True)

   if gVec0[0] == True:
      ax1 = figs.add_subplot(331)
      subplots.append(ax1)
      r = minerals.index(clickedS.get())
      ax1.plot(lamMineral[r], mMineral[r], label="{}".format(clickedS.get()))
      q = minerals.index(clickedM.get())
      ax1.plot(lamMineral[q], mMineral[q], label="{}".format(clickedM.get()))
      ax1.axvline(x=.440, color="r")
      ax1.axvline(x=.520, color="r")
      ax1.set_xlim((0.4, .6))
      ax1.set_ylim((0, 3))
      ax1.set_xlabel('Wavelength (microns)')
      ax1.set_ylabel('Refractive Index')
      ax1.legend()
      ax1.set_title(' Refractive Index vs. Wavelength')

   #If single Particle Size
   if isfloat(a0) == True:
      a = a0
      aQ = []
      aI = []
      sigma=[]

      #If single Wavelength
      if isfloat(lambda0) == True:
         lam = lambda0
         k = 2 * np.pi / lam  # wave number
         x = (a / 2) * k
         m_sphere = m_sphere0
         [s] = np.where(lamMineral[6] == lam)
         if len(s) == 0:
            newlambda = find_nearest(lamMineral[6], lam)
            [s] = np.where(lamMineral[6] == newlambda)

         mre = m_aerosol0
         geometric_cross_section = np.pi * (a / 2) ** 2
         print(mre)
         print(m_sphere)
         m = m_sphere / mre
         if clickedM.get() == 'No Mineral':
            m = m_sphere
            qext, qsca, qback, g = miepython.mie(m, x)
         else:
            qext, qsca, qback, g = miepython.ez_mie(m_sphere, a, lam, mre)
         mu = np.cos(theta * np.pi / 180)
         S1, S2 = miepython.mie_S1_S2(m, x, mu)

         S11 = np.abs(S2) ** 2
         S12 = np.abs(S1) ** 2
         c = 1
         P11 = c * S11
         P12 = c * S12
         I = S11
         Q = S12
         sig = qsca * geometric_cross_section

         if gVec0[1] == True:
            ax2 = figs.add_subplot(332)
            subplots.append(ax2)
            ax2.scatter(a, sig)
            ax2.set_xlabel("Particle Diameter ($\mu m$)")
            ax2.set_ylabel("Scattering Cross Section ($ cm^2$)")
            ax2.set_title("Scattering Cross Section")
         if gVec0[3] == True:
            ax4 = figs.add_subplot(334, projection='polar')
            subplots.append(ax4)
            ax4.plot(theta / 180 * np.pi, I)
            ax4.axvline(eVec0[0] * np.pi / 180, color="r")  # it is in radiants
            ax4.axvline(eVec0[1] * np.pi / 180, color="r")
         if gVec0[4] == True:
            ax5 = figs.add_subplot(335)
            subplots.append(ax5)
            ax5.plot(theta, I)
            ax5.set_xlim((0, 181))
            ax5.axvline(x=eVec0[0], color="r")
            ax5.axvline(x=eVec0[1], color="r")
            ax5.set_xlabel("Scattering Angle (α°)")
            ax5.set_ylabel("Scattered Light")
            ax5.set_title("I")
         if gVec0[5] == True:
            ax6 = figs.add_subplot(336)
            subplots.append(ax6)
            ax6.plot(theta, Q)
            ax6.set_xlim((0, 181))
            ax6.axvline(x=eVec0[0], color="r")
            ax6.axvline(x=eVec0[1], color="r")
            ax6.set_xlabel("Scattering Angle (α°)")
            ax6.set_ylabel("Q")
            ax6.set_title("Q")

         # plt.show()
         def rawI(x):
            #tt = np.arange(-181, 181, 1)
            tt = np.arange(eVec0[0], eVec0[1]+1, 1)
            [k] = np.where(tt == x)
            if len(k) == 0:
               newTheta = find_nearest(tt, x)
               [k] = np.where(tt == newTheta)
            y = float(I[k])
            return y

         def rawQ(x):
            #tt = np.arange(-181, 181, 1)
            tt = np.arange(eVec0[0], eVec0[1]+1, 1)
            [k] = np.where(tt == x)
            if len(k) == 0:
               newTheta = find_nearest(tt, x)
               [k] = np.where(tt == newTheta)
            y = float(Q[k])
            return y

         rI = 0
         rQ = 0
         tt = np.arange(eVec0[0], eVec0[1]+0.1, 0.1)
         for f in tt:
            fl = eVec0[3]*1000  # Focal Length (200000 microns)
            midTheta = (max(tt) + min(tt)) / 2  # Mid Point Theta (rad)
            deltaTheta = np.abs(f - midTheta) * np.pi / 180  # Maximum delta (rad)
            windowR = eVec0[2]*1000   # Window radius
            h = fl * np.tan(deltaTheta)
            d = np.sqrt((windowR ** 2) - (h ** 2))
            weight = 0.2 * np.arctan(d / fl) * np.pi / 180

            def WrawQ(x):
               return rawQ(x) * weight

            resaQ=WrawQ(f)
            rQ = rQ + resaQ

            def WrawI(x):
               return rawI(x) * weight

            resaI=WrawI(f)
            rI = rI + resaI

         if gVec0[2] == True:
            ax3 = figs.add_subplot(333)
            subplots.append(ax3)
            ax3.scatter(a, rQ, label="Q - Perpendicular Intensity")
            ax3.scatter(a, rI, label="I - Parallel Intensity")
            ax3.set_xlabel("Particle Size (microns)")
            ax3.set_ylabel("Detected Intensity")
            ax3.set_title("Intensity vs Particle Size")
            ax3.legend()

         ratioParticle = (rQ - rI) / (rQ + rI)
         if gVec0[6] == True:
            ax7 = figs.add_subplot(337)
            subplots.append(ax7)
            ax7.scatter(ratioParticle, rQ / rI, label="{}".format(lam))
            ax7.set_xlabel("Ratio")
            ax7.set_ylabel("resQ/resI")
            ax7.set_title("Simulated Lab Result 1")
            ax7.legend()

         if gVec0[7] == True:
            ax8 = figs.add_subplot(338)
            subplots.append(ax8)
            ax8.scatter(rQ, ratioParticle, label="{}".format(lam))
            ax8.set_xlabel("resQ")
            ax8.set_ylabel("Ratio")
            ax8.set_title("Simulated Lab Result 2")
            ax8.legend()
         if gVec0[8] == True:
            ax9 = figs.add_subplot(339)
            subplots.append(ax9)
            ax9.scatter(rQ + rI, ratioParticle, label="{}".format(lam))
            ax9.set_xlabel("resQ+resI")
            ax9.set_ylabel("Ratio")
            ax9.set_title("Simulated Lab Result 3")
            ax9.legend()

      #If range of Wavelength
      elif isfloat(lambda0) == False:
         resI = []
         resQ = []
         ratioParticle = []
         resPlus = []
         resFrac = []
         for i in range(len(lambda0)):
            lam = lambda0[i]
            k = 2 * np.pi / lam  # wave number
            x = (a) * k / 2
            m_sphere = m_sphere0[i]
            # print(m_sphere)
            [s] = np.where(lamMineral[6] == lam)
            if len(s) == 0:
               newlambda = find_nearest(lamMineral[6], lam)
               [s] = np.where(lamMineral[6] == newlambda)
            mre = m_aerosol0[i]
            geometric_cross_section = np.pi * (a / 2) ** 2
            m = m_sphere / mre
            if clickedM.get() == 'No Mineral':
               m = m_sphere
               qext, qsca, qback, g = miepython.mie(m, x)
            else:
               qext, qsca, qback, g = miepython.ez_mie(m_sphere, a, lam, mre)
            mu = np.cos(theta * np.pi / 180)
            S1, S2 = miepython.mie_S1_S2(m, x, mu)
            # qext, qsca, qback, g = miepython.ez_mie(m_sphere, a, lam, mre)

            S11 = np.abs(S2) ** 2
            S12 = np.abs(S1) ** 2
            S33 = (S2 * S1.conjugate()).real
            S34 = (S2 * S1.conjugate()).imag
            c = 1
            P11 = c * S11
            P12 = c * S12
            # I=P11+P12
            # Q=P12+P11/(P11+P12)
            I = S11
            Q = S12
            sig = qsca * geometric_cross_section
            sigma.append(sig)
            aI.append(I)
            aQ.append(Q)

            def rawI(x):
               tt = np.arange(eVec0[0], eVec0[1]+1, 1)
               [k] = np.where(tt == x)
               if len(k) == 0:
                  newTheta = find_nearest(tt, x)
                  [k] = np.where(tt == newTheta)
               y = float(I[k])
               return y

            #resI = []

            def rawQ(x):
               tt = np.arange(eVec0[0], eVec0[1]+1, 1)
               [k] = np.where(tt == x)
               if len(k) == 0:
                  newTheta = find_nearest(tt, x)
                  [k] = np.where(tt == newTheta)
               y = float(Q[k])
               return y

            #resQ = []
            #ratioParticle = []
            #resPlus = []
            #resFrac = []

            rI = 0
            rQ = 0
            rP = 0
            rPlus = 0
            rFrac = 0
            tt = np.arange(eVec0[0], eVec0[1]+0.1, 0.1)
            for f in tt:
               fl = eVec0[3]*1000  # Focal Length (200000 microns)
               midTheta = (max(tt) + min(tt)) / 2  # Mid Point Theta (rad)
               deltaTheta = np.abs(f - midTheta) * np.pi / 180  # Maximum delta (rad)
               windowR = eVec0[2]*1000 # Window radius
               h = fl * np.tan(deltaTheta)
               d = np.sqrt((windowR ** 2) - (h ** 2))
               weight = 0.2 * np.arctan(d / fl) * np.pi / 180

               def WrawQ(x):
                  return rawQ(x) * weight

               resaQ= WrawQ(f)
               rQ = rQ + resaQ

               def WrawI(x):
                  return rawI(x) * weight

               resaI=WrawI(f)
               rI = rI + resaI
               rP = (rQ - rI) / (rQ + rI)
               rPlus = (rQ + rI)
               rFrac = (rQ / rI)
            resQ.append(rQ)
            resI.append(rI)
            ratioParticle.append(rP)
            resPlus.append(rPlus)
            resFrac.append(rFrac)

         if gVec0[1] == True:
            ax2 = figs.add_subplot(332)
            subplots.append(ax2)
            for i in range(len(lambda0)):
               ax2.scatter(a, sigma[i], label="{}".format(lambda0[i]))
            ax2.set_xlabel("Particle Diameter ($\mu m$)")
            ax2.set_ylabel("Scattering Cross Section ($ cm^2$)")
            ax2.set_title("Scattering Cross Section")
            ax2.legend(loc='upper left',ncol=2)
         if gVec0[3] == True:
            ax4 = figs.add_subplot(334, projection='polar')
            subplots.append(ax4)
            for i in range(len(lambda0)):
               ax4.plot(theta / 180 * np.pi, aI[i])
            ax4.axvline(eVec0[0] * np.pi / 180, color="r")  # it is in radiants
            ax4.axvline(eVec0[1] * np.pi / 180, color="r")
         if gVec0[4] == True:
            ax5 = figs.add_subplot(335)
            subplots.append(ax5)
            for i in range(len(lambda0)):
               ax5.plot(theta, aI[i])
            ax5.set_xlim((0, 181))
            ax5.axvline(x=eVec0[0], color="r")
            ax5.axvline(x=eVec0[1], color="r")
            ax5.set_xlabel("Scattering Angle (α°)")
            ax5.set_ylabel("Scattered Light")
            ax5.set_title("I")
         if gVec0[5] == True:
            ax6 = figs.add_subplot(336)
            subplots.append(ax6)
            for i in range(len(lambda0)):
               ax6.plot(theta, aQ[i])
            ax6.set_xlim((0, 181))
            ax6.axvline(x=eVec0[0], color="r")
            ax6.axvline(x=eVec0[1], color="r")
            ax6.set_xlabel("Scattering Angle (α°)")
            ax6.set_ylabel("Q/I")
            ax6.set_title("Q")

         if gVec0[2] == True:
            ax3 = figs.add_subplot(333)
            subplots.append(ax3)
            for i in range(len(lambda0)):
               ax3.scatter(a, resQ[i], label="Q - {}".format(lambda0[i]))
               ax3.scatter(a, resI[i], label="I - {}".format(lambda0[i]))
            ax3.set_xlabel("Particle Size (microns)")
            ax3.set_ylabel("Detected Intensity")
            ax3.set_title("Intensity vs Particle Size")
            ax3.legend(ncol=2)

         if gVec0[6] == True:
            ax7 = figs.add_subplot(337)
            subplots.append(ax7)
            for i in range(len(lambda0)):
               ax7.scatter(ratioParticle[i], resFrac[i], marker=markers[i % len(markers)], label="{}".format(lambda0[i]))
            ax7.set_xlabel("Ratio")
            ax7.set_ylabel("resQ/resI")
            ax7.set_title("Simulated Lab Result 1")
            ax7.legend()

         if gVec0[7] == True:
            ax8 = figs.add_subplot(338)
            subplots.append(ax8)
            for i in range(len(lambda0)):
               ax8.scatter(resQ[i], ratioParticle[i], marker=markers[i % len(markers)], label="{}".format(lambda0[i]))
            ax8.set_xlabel("resQ")
            ax8.set_ylabel("Ratio")
            ax8.set_title("Simulated Lab Result 2")
            ax8.legend()

         if gVec0[8] == True:
            ax9 = figs.add_subplot(339)
            subplots.append(ax9)
            for i in range(len(lambda0)):
               ax9.scatter(resPlus[i], ratioParticle[i], marker=markers[i % len(markers)], label="{}".format(lambda0[i]))
            ax9.set_xlabel("resQ+resI")
            ax9.set_ylabel("Ratio")
            ax9.set_title("Simulated Lab Result 3")
            ax9.legend()


            # plt.show()

   #If range of Particle Size
   elif isfloat(a0) == False:

      #If single wavelength
      if isfloat(lambda0) == True:
         sigma = []
         aQ = []
         aI = []
         for j in range(len(a0)):
            a = a0[j]
            lam = lambda0
            k = 2 * np.pi / lam  # wave number
            x = (a) * k / 2
            m_sphere = m_sphere0
            [s] = np.where(lamMineral[6] == lam)
            if len(s) == 0:
               newlambda = find_nearest(lamMineral[6], lam)
               [s] = np.where(lamMineral[6] == newlambda)
            mre = m_aerosol0
            geometric_cross_section = np.pi * (a / 2) ** 2
            m = m_sphere / mre
            if clickedM.get() == 'No Mineral':
               m = m_sphere
               qext, qsca, qback, g = miepython.mie(m, x)
            else:
               qext, qsca, qback, g = miepython.ez_mie(m_sphere, a, lam, mre)
            mu = np.cos(theta * np.pi / 180)
            S1, S2 = miepython.mie_S1_S2(m, x, mu)
            # qext, qsca, qback, g = miepython.ez_mie(m_sphere, a, lam, mre)
            S11 = np.abs(S2) ** 2
            S12 = np.abs(S1) ** 2
            c = 1
            P11 = c * S11
            P12 = c * S12

            I = S11
            Q = S12
            # print(I)
            sig = qsca * geometric_cross_section
            sigma.append(sig)
            aI.append(I)
            aQ.append(Q)

            # if gVec0[2]==True:
            # plt.subplot(3, 3, 3)


            # plt.show()
         if gVec0[1] == True:
            ax2 = figs.add_subplot(332)
            subplots.append(ax2)
            ax2.plot(a0, sigma)
            ax2.set_xlabel("Particle Diameter ($\mu m$)")
            ax2.set_ylabel("Scattering Cross Section ($ cm^2$)")
            ax2.set_title("Scattering Cross Section")
            # plt.legend(loc='upper left')

         def rawI(x):
            tt = np.arange(eVec0[0], eVec0[1]+1, 1)
            [k] = np.where(tt == x)
            if len(k) == 0:
               newTheta = find_nearest(tt, x)
               [k] = np.where(tt == newTheta)
            y = float(aI[e][k])
            return y

         resI = []

         def rawQ(x):
            tt = np.arange(eVec0[0], eVec0[1]+1, 1)
            [k] = np.where(tt == x)
            if len(k) == 0:
               newTheta = find_nearest(tt, x)
               [k] = np.where(tt == newTheta)
            y = float(aQ[e][k])
            return y

         resQ = []
         ratioParticle = []
         resPlus = []
         resFrac = []
         for e in range(0, len(aQ)):
            rQ = 0
            rI = 0
            rP = 0
            rPlus = 0
            rFrac = 0
            tt = np.arange(eVec0[0], eVec0[1]+0.1, 0.1)
            for f in tt:
               fl = eVec0[3]*1000  # Focal Length (200000 microns)
               midTheta = (max(tt) + min(tt)) / 2  # Mid Point Theta (rad)
               deltaTheta = np.abs(f - midTheta) * np.pi / 180  # Maximum delta (rad)
               windowR = eVec0[2]*1000# Window radius
               h = fl * np.tan(deltaTheta)
               d = np.sqrt((windowR ** 2) - (h**2))
               weight = 0.2*np.arctan(d/fl)*np.pi/180

               def WrawQ(x):
                  return rawQ(x) * weight

               resaQ=WrawQ(f)
               rQ = rQ + resaQ

               def WrawI(x):
                  return rawI(x) * weight

               resaI=WrawI(f)
               rI = rI + resaI
            rP = (rQ - rI) / (rQ + rI)
            rPlus = (rQ + rI)
            rFrac = (rQ / rI)
            resQ.append(rQ)
            resI.append(rI)
            ratioParticle.append(rP)
            resPlus.append(rPlus)
            resFrac.append(rFrac)

         if gVec0[3] == True:
            ax4 = figs.add_subplot(334, projection='polar')
            subplots.append(ax4)
            for j in range(len(a0)):
               ax4.plot(theta / 180 * np.pi, aI[j])
            ax4.axvline(168 * np.pi / 180, color="r")  # it is in radiants
            ax4.axvline(173 * np.pi / 180, color="r")

         if gVec0[4] == True:
            ax5 = figs.add_subplot(335)
            subplots.append(ax5)
            for j in range(len(a0)):
               ax5.plot(theta, aI[j])
            ax5.set_xlim((0, 181))
            ax5.axvline(x=168, color="r")
            ax5.axvline(x=173, color="r")
            ax5.set_xlabel("Scattering Angle (α°)")
            ax5.set_ylabel("Scattered Light")
            ax5.set_title("I")

         if gVec0[5] == True:
            ax6 = figs.add_subplot(336)
            subplots.append(ax6)
            for j in range(len(a0)):
               ax6.plot(theta, aQ[j])
            ax6.set_xlim((0, 181))
            ax6.axvline(x=eVec0[0], color="r")
            ax6.axvline(x=eVec0[1], color="r")
            ax6.set_xlabel("Scattering Angle (α°)")
            ax6.set_ylabel("Q/I")
            ax6.set_title("Q")

         if gVec0[2] == True:
            ax3 = figs.add_subplot(333)
            subplots.append(ax3)
            ax3.plot(a0, resQ, label="Q - Perpendicular Intensity")
            ax3.plot(a0, resI, label="I - Parallel Intensity")
            ax3.set_xlabel("Particle Size (microns)")
            ax3.set_ylabel("Detected Intensity")
            ax3.set_title("Intensity vs Particle Size")
            ax3.legend()
         # ratioParticle=(resQ-resI)/(resQ+resI)
         # print(ratioParticle)
         if gVec0[6] == True:
            ax7 = figs.add_subplot(337)
            subplots.append(ax7)
            for j in range(len(a0)):
               sc7=ax7.scatter(ratioParticle[j], resFrac[j], label="{}".format(lam), c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
            ax7.set_xlabel("Ratio")
            ax7.set_ylabel("resQ/resI")
            ax7.set_title("Simulated Lab Result 1")
            cbar7=plt.colorbar(sc7)
            cbar7.ax.set_title('a0')
         if gVec0[7] == True:
            ax8 = figs.add_subplot(338)
            subplots.append(ax8)
            for j in range(len(a0)):
               sc8=ax8.scatter(resQ[j], ratioParticle[j], label="{}".format(lam), c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
            ax8.set_xlabel("resQ")
            ax8.set_ylabel("Ratio")
            ax8.set_title("Simulated Lab Result 2")
            cbar8=plt.colorbar(sc8)
            cbar8.ax.set_title('a0')

         if gVec0[8] == True:
            ax9 = figs.add_subplot(339)
            subplots.append(ax9)
            for j in range(len(a0)):
               sc9=ax9.scatter(resPlus[j], ratioParticle[j], label="{}".format(lam), c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
            ax9.set_xlabel("resQ+resI")
            ax9.set_ylabel("Ratio")
            ax9.set_title("Simulated Lab Result 3")
            cbar9=plt.colorbar(sc9)
            cbar9.ax.set_title('a0')

      #If range of wavelength
      elif isfloat(lambda0) == False:
         allQ = []
         allI = []
         allS = []
         labelQ = []
         labelI = []
         labLam=[]
         for i in range(len(lambda0)):
            lam = lambda0[i]
            labelQ.append("Q - {}".format(lam))
            labelI.append("I - {}".format(lam))
            labLam.append("{}".format(lam))
            sigma = []
            aQ = []
            aI = []
            for j in range(len(a0)):
               a = a0[j]
               k = 2 * np.pi / lam  # wave number
               x = (a) * k / 2
               m_sphere = m_sphere0[i]
               [s] = np.where(lamMineral[6] == lam)
               if len(s) == 0:
                  newlambda = find_nearest(lamMineral[6], lam)
                  [s] = np.where(lamMineral[6] == newlambda)
               mre = m_aerosol0[i]
               geometric_cross_section = np.pi * (a / 2) ** 2
               m = m_sphere / mre
               if clickedM.get() == 'No Mineral':
                  m = m_sphere
                  qext, qsca, qback, g = miepython.mie(m, x)
               else:
                  qext, qsca, qback, g = miepython.ez_mie(m_sphere, a, lam, mre)
               mu = np.cos(theta * np.pi / 180)
               S1, S2 = miepython.mie_S1_S2(m, x, mu)

               # qext, qsca, qback, g = miepython.ez_mie(m_sphere, a, lam, mre)
               S11 = np.abs(S2) ** 2
               S12 = np.abs(S1) ** 2
               c = 1
               P11 = c * S11
               P12 = c * S12

               I = S11
               Q = S12
               sig = qsca * geometric_cross_section
               sigma.append(sig)
               aI.append(I)
               aQ.append(Q)

               # if gVec0[2]==True:
               # plt.subplot(3, 3, 3)
            allI.append(aI)
            allQ.append(aQ)
            allS.append(sigma)

         def rawI(x):
            tt = np.arange(168, 174, 1)
            [k] = np.where(tt == x)
            if len(k) == 0:
               newTheta = find_nearest(tt, x)
               [k] = np.where(tt == newTheta)
            y = float(allI[e][l][k])
            return y

         def rawQ(x):
            tt = np.arange(168, 174, 1)
            [k] = np.where(tt == x)
            if len(k) == 0:
               newTheta = find_nearest(tt, x)
               [k] = np.where(tt == newTheta)
            y = float(allQ[e][l][k])
            return y

         aresQ = []
         aresI = []
         aratioParticle = []
         aresPlus = []
         aresFrac = []
         for l in range(len(a0)):
            resQ = []
            resI = []
            ratioParticle = []
            resPlus = []
            resFrac = []
            for e in range(0, len(allQ)):
               rQ = 0
               rI = 0
               rP = 0
               rPlus = 0
               rFrac = 0
               tt = np.arange(eVec0[0], eVec0[1]+0.1, 0.1)
               for f in tt:
                  fl = eVec0[3]*1000  # Focal Length (200000 microns)
                  midTheta = (max(tt) + min(tt)) / 2  # Mid Point Theta (rad)
                  deltaTheta = np.abs(f - midTheta) * np.pi / 180  # Maximum delta (rad)
                  windowR =eVec0[2]*1000   # Window radius
                  h = fl * np.tan(deltaTheta)
                  d = np.sqrt((windowR ** 2) - (h ** 2))
                  weight = 0.2 * np.arctan(d / fl) * np.pi / 180

                  def WrawQ(x):
                     return rawQ(x) * weight

                  resaQ=WrawQ(f)
                  rQ = rQ + resaQ

                  def WrawI(x):
                     return rawI(x) * weight

                  resaI=WrawI(f)
                  rI = rI + resaI
               rP = (rQ - rI) / (rQ + rI)
               rPlus = (rQ + rI)
               rFrac = (rQ / rI)
               resQ.append(rQ)
               resI.append(rI)
               ratioParticle.append(rP)
               resPlus.append(rPlus)
               resFrac.append(rFrac)
            aresQ.append(resQ)
            aresI.append(resI)
            aratioParticle.append(ratioParticle)
            aresPlus.append(resPlus)
            aresFrac.append(resFrac)

         if gVec0[1] == True:
            ax2 = figs.add_subplot(332)
            subplots.append(ax2)
            for i in range(len(lambda0)):
               ax2.plot(a0, allS[i], label="{}".format(lambda0[i]))
            ax2.set_xlabel("Particle Diameter ($\mu m$)")
            ax2.set_ylabel("Scattering Cross Section ($ cm^2$)")
            ax2.set_title("Scattering Cross Section")
            ax2.legend(loc='upper left')

         if gVec0[3] == True:
            ax4 = figs.add_subplot(334, projection='polar')
            subplots.append(ax4)
            for i in range(len(lambda0)):
               for j in range(len(a0)):
                  ax4.plot(theta / 180 * np.pi, allI[i][j])
            ax4.axvline(eVec0[0] * np.pi / 180, color="r")  # it is in radiants
            ax4.axvline(eVec0[1] * np.pi / 180, color="r")

         if gVec0[4] == True:
            ax5 = figs.add_subplot(335)
            subplots.append(ax5)
            for i in range(len(lambda0)):
               for j in range(len(a0)):
                  ax5.plot(theta, allI[i][j])
            ax5.set_xlim((0, 181))
            ax5.axvline(x=eVec0[0], color="r")
            ax5.axvline(x=eVec0[1], color="r")
            ax5.set_xlabel("Scattering Angle (α°)")
            ax5.set_ylabel("Scattered Light")
            ax5.set_title("I")

         if gVec0[5] == True:
            ax6 = figs.add_subplot(336)
            subplots.append(ax6)
            for i in range(len(lambda0)):
               for j in range(len(a0)):
                  ax6.plot(theta, allQ[i][j])
            ax6.set_xlim((0, 181))
            ax6.axvline(x=eVec0[0], color="r")
            ax6.axvline(x=eVec0[0], color="r")
            ax6.set_xlabel("Scattering Angle (α°)")
            ax6.set_ylabel("Q/I")
            ax6.set_title("Q")

         if gVec0[2] == True:
            ax3 = figs.add_subplot(333)
            subplots.append(ax3)
            ax3.plot(a0, aresQ, label=labelQ)
            ax3.plot(a0, aresI, label=labelI)
            ax3.set_xlabel("Particle Size (microns)")
            ax3.set_ylabel("Detected Intensity")
            ax3.set_title("Intensity vs Particle Size")
            ax3.legend(ncol=2)
         if gVec0[6] == True:
            ax7 = figs.add_subplot(337)
            subplots.append(ax7)
            for i in range(len(lambda0)):
               for j in range(len(a0)):
                  if j == 0:  # Only label the first point for each lambda0
                     sc7 = ax7.scatter(aratioParticle[j][i], aresFrac[j][i], marker=markers[i % len(markers)], label=labLam[i], c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
                  else:
                     sc7 = ax7.scatter(aratioParticle[j][i], aresFrac[j][i], marker=markers[i % len(markers)], c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
            ax7.set_xlabel("Ratio")
            ax7.set_ylabel("resQ/resI")
            ax7.set_title("Simulated Lab Result 1")
            ax7.legend()
            cbar7 = plt.colorbar(sc7)
            cbar7.ax.set_title('a0')

         if gVec0[7] == True:
            ax8 = figs.add_subplot(338)
            subplots.append(ax8)
            for i in range(len(lambda0)):
               for j in range(len(a0)):
                  if j == 0:  # Only label the first point for each lambda0
                     sc8 = ax8.scatter(aresQ[j][i],aratioParticle[j][i], marker=markers[i % len(markers)], label=labLam[i], c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
                  else:
                     sc8=ax8.scatter(aresQ[j][i], aratioParticle[j][i], marker=markers[i % len(markers)], c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
            ax8.set_xlabel("resQ")
            ax8.set_ylabel("Ratio")
            ax8.set_title("Simulated Lab Result 2")
            ax8.legend()
            cbar8=plt.colorbar(sc8)
            cbar8.ax.set_title('a0')
         if gVec0[8] == True:
            ax9 = figs.add_subplot(339)
            subplots.append(ax9)
            for i in range(len(lambda0)):
               for j in range(len(a0)):
                  if j == 0:
                     sc9=ax9.scatter(aresPlus[j][i],aratioParticle[j][i], marker=markers[i % len(markers)], label=labLam[i], c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
                  else:
                     sc9 = ax9.scatter(aresPlus[j][i], aratioParticle[j][i], marker=markers[i % len(markers)], c=a0[j], vmin=min(a0), vmax=max(a0), cmap=cm)
            ax9.set_xlabel("resQ+resI")
            ax9.set_ylabel("Ratio")
            ax9.set_title("Simulated Lab Result 3")
            ax9.legend()
            cbar9=plt.colorbar(sc9)
            cbar9.ax.set_title('a0')
   figs.tight_layout()

   #  plt.show()
   canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
   saveB = tkinter.Button(newWindow, text="Save", command=saveCallBack)
   saveB.pack()
def saveCallBack():
   # Save the full figure...
   figs.savefig("full_figure.png")
   lenax = np.count_nonzero(gVector)

   for i in range(lenax):
      extent = subplots[i].get_window_extent().transformed(figs.dpi_scale_trans.inverted())
      figs.savefig('ax{}_figure.png'.format(i), bbox_inches=extent.expanded(1.3, 1.3))

def goCallBack():
   global gVector, subplots
   sVector = [sMineral, aS, sWavelen,sAerosol]
   #print(sVector)
   gVector = [CheckVar1.get(), CheckVar2.get(), CheckVar3.get(), CheckVar4.get(), CheckVar5.get(), CheckVar6.get(), CheckVar7.get(), CheckVar8.get(), CheckVar9.get()]
   eVector=[float(anMinS.get()),float(anMaxS.get()),float(winRS.get()),float(focDS.get())]
   #print(eVector)
   # newWindow = Toplevel(root)
   # newWindow.title("Mie Scattering Simulation Results")
   # newWindow.geometry("500x500")
   # Label(newWindow, text=clickedM.get()).pack()
   # f = plt.Figure(figsize=(16, 8), dpi=100)
   # canvas = FigureCanvasTkAgg(f, newWindow)
   subplots = []
   allG(sVector[0], sVector[1], sVector[2], sVector[3], gVector, eVector)
   # canvas.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=True)
   saveB.pack()

B = tkinter.Button(frame4, text ="Run", command = goCallBack)
saveB = tkinter.Button(frame4, text="Save", command=saveCallBack)


CheckVar1 = IntVar()
CheckVar2 = IntVar()
CheckVar3 = IntVar()
CheckVar4 = IntVar()
CheckVar5 = IntVar()
CheckVar6 = IntVar()
CheckVar7 = IntVar()
CheckVar8 = IntVar()
CheckVar9 = IntVar()
C1 = Checkbutton(frame3, text = "Refractive Index vs Wavelen", variable = CheckVar1, onvalue = 1, offvalue = 0)
C2 = Checkbutton(frame3, text = "Scattering Cross Section", variable = CheckVar2,onvalue = 1, offvalue = 0)
C3 = Checkbutton(frame3, text = "Intensity vs Particle Size", variable = CheckVar3, onvalue = 1, offvalue = 0)
C4 = Checkbutton(frame3, text = "Polar Representation: I", variable = CheckVar4,onvalue = 1, offvalue = 0)
C5 = Checkbutton(frame3, text = "Stokes Parameter: I", variable = CheckVar5, onvalue = 1, offvalue = 0)
C6 = Checkbutton(frame3, text = "Stokes Parameter: Q", variable = CheckVar6,onvalue = 1, offvalue = 0)
C7 = Checkbutton(frame3, text = "Lab Sim: S/P vs Ratio", variable = CheckVar7, onvalue = 1, offvalue = 0)
C8 = Checkbutton(frame3, text = "Lab Sim: Ratio vs Speak", variable = CheckVar8,onvalue = 1, offvalue = 0)
C9 = Checkbutton(frame3, text = "Lab Sim: Ratio vs Sum", variable = CheckVar9, onvalue = 1, offvalue = 0)

C1.pack()
C2.pack()
C3.pack()
C4.pack()
C5.pack()
C6.pack()
C7.pack()
C8.pack()
C9.pack()
B.pack()

root.mainloop()
# **ARIADNE**  
*(spectrAl eneRgy dIstribution bAyesian moDel averagiNg fittEr)*  

### Now Extended for **Binary SED Fitting**

---

## **Introduction**

**ARIADNE** is a spectral energy distribution (SED) fitting tool based on nested sampling algorithms.  
The original version, developed by [James Vines](https://github.com/jvines/astroARIADNE), supports multiple stellar atmosphere models — such as **Phoenix v2**, **BT-Settl**, **BT-Cond**, **BT-NextGen**, **Castelli & Kurucz (2004)**, and **Kurucz (1993)** to combine model-dependent posterior distributions.

This modified version, extends ARIADNE to fit **binary systems’ SEDs**.  
Two additional models have been integrated:
- **Koester model** — for white dwarfs  
- **TMAP model** — for hot subdwarfs  

These models can be combined with the main-sequence star model to fit composite binary SEDs.

---

## **Installation**

1. First, install all dependencies following the original **astroARIADNE** installation guide:  
   👉 [astroARIADNE Installation Guide](https://github.com/jvines/astroARIADNE/blob/master/README.md)

2. Clone this repository:  
   ```bash
   git clone https://github.com/ymkuan/Ariadnes_binary.git

3. After downloading and installation, rename the cloned folder to:
   astroAriadne_binary
 
## In order to plot the models, you have to download them first:
But note that plotting the SED model is optional. You can run the code without
them!

| Model        | Link           |
| ------------- |:-------------:|
| Phoenix v2      | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/> |
| Phoenix v2   wavelength file   | <ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits> |
| BT-Models      | <http://svo2.cab.inta-csic.es/theory/newov2/>  |
| Castelli & Kurucz | <http://ssb.stsci.edu/cdbs/tarfiles/synphot3.tar.gz>      |
| Kurucz 1993 | <http://ssb.stsci.edu/cdbs/tarfiles/synphot4.tar.gz>  |
| Koester WD  | <https://svo2.cab.inta-csic.es//theory/newov2/temp/models/tar/models_1762818096.tgz>  |
| TMAP        | <https://drive.google.com/file/d/1yKHB3uB7QUyhhkwa1hNMuWazxjLkLAjN/view>

The wavelength file for the Phoenix model has to be placed in the root folder
of the PHOENIXv2 models.

For the code to find these models, you have to place them somewhere in your
computer as follows:

```
Models_Dir  
│
└───BTCond
│   │
│   └───CIFIST2011
│   
└───BTNextGen
│	 │
│	 └───AGSS2009
│
└───BTSettl
│	 │
│	 └───AGSS2009
│
└───Castelli_Kurucz
│	 │
│	 └───ckm05
│	 │
│	 └───ckm10
│	 │
│	 └───ckm15
│	 │
│	 └───ckm20
│	 │
│	 └───ckm25
│	 │
│	 └───ckp00
│	 │
│	 └───ckp02
│	 │
│	 └───ckp05
│
└───Kurucz
│	 │
│	 └───km01
│	 │
│	 └───km02
│	 │
│	 └───km03
│	 │
│	 └───km05
│	 │
│	 └───km10
│	 │
│	 └───km15
│	 │
│	 └───km20
│	 │
│	 └───km25
│	 │
│	 └───kp00
│	 │
│	 └───kp01
│	 │
│	 └───kp02
│	 │
│	 └───kp03
│	 │
│	 └───kp05
│	 │
│	 └───kp10
│
└───PHOENIXv2
	 │
     └─── WAVE_PHOENIX-ACES-AGSS-COND-2011.fits
	 └───Z-0.0
	 │
	 └───Z-0.5
	 │
	 └───Z-1.0
	 │
	 └───Z-1.5
	 │
	 └───Z-2.0
	 │
	 └───Z+0.5
	 │
	 └───Z+1.0
└───KoesterWD
	 │
     └─── koester
	        └───daxxx.dk.da.fits
└───Subdwarf_TMAP
	 │
     └─── sp
	        └───T20000_logg4.50.csv
			└───......
### Notes:
- The Phoenix v2 models with alpha enhancements are unused
- BT-models are BT-Settl, BT-Cond, and BT-NextGen

# How to use?

see example.py
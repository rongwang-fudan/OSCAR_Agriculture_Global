"""
Copyright: IIASA (International Institute for Applied Systems Analysis), 2016-2018; CEA (Commissariat a L'Energie Atomique) & UVSQ (Universite de Versailles et Saint-Quentin), 2016
Contributor(s): Thomas Gasser (gasser@iiasa.ac.at)

This software is a computer program whose purpose is to simulate the behavior of the Earth system, with a specific but not exclusive focus on anthropogenic climate change.

This software is governed by the CeCILL license under French law and abiding by the rules of distribution of free software.  You can use, modify and/ or redistribute the software under the terms of the CeCILL license as circulated by CEA, CNRS and INRIA at the following URL "http://www.cecill.info". 

As a counterpart to the access to the source code and rights to copy, modify and redistribute granted by the license, users are provided only with a limited warranty and the software's author, the holder of the economic rights, and the successive licensors have only limited liability. 

In this respect, the user's attention is drawn to the risks associated with loading, using, modifying and/or developing or reproducing the software by the user in light of its specific status of free software, that may mean that it is complicated to manipulate, and that also therefore means that it is reserved for developers and experienced professionals having in-depth computer knowledge. Users are therefore encouraged to load and test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or data to be ensured and,  more generally, to use and operate it in the same conditions as regards security. 

The fact that you are presently reading this means that you have had knowledge of the CeCILL license and that you accept its terms.
"""


##################################################
##################################################
##################################################


import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin,fsolve
from scipy.special import gammainc
from matplotlib.font_manager import FontProperties


##################################################
#   1. OPTIONS
##################################################

p = 6                                   # time step (p-th year; must be >4)
fC = 1                                  # carbon feedback (0 or 1)
fT = 1                                  # climate feedback (0 or 1)
dty = np.float32                        # precision (float32 or float64)

PI_1750 = True                         # if False simulates the 1700-1750 period
ind_final = 500                         # ending year of run (+1700)
ind_attrib = 0                          # starting year of attribution

attrib_DRIVERS = 'producers'            # [deprecated]
attrib_FEEDBACKS = 'emitters'           # [deprecated]
attrib_ELUCdelta = 'causal'             # [deprecated]
attrib_ELUCampli = 'causal'             # [deprecated]

mod_regionI = 'Houghton'                # SRES4 | SRES11 | RECCAP* | Raupach* | Houghton | IMACLIM | Kyoto | RCP5 | RCP10*
mod_regionJ = 'RCP5'                    # SRES4 | SRES11 | RECCAP* | Raupach* | Houghton | IMACLIM | Kyoto | RCP5 | RCP10*
mod_sector = ''                         # '' | Time | TimeRCP
mod_kindFF = 'one'                      # one | CDIAC
mod_kindLUC = 'one'                     # one | all
mod_kindGHG = 'one'                     # one | RCP
mod_kindCHI = 'one'                     # one | all
mod_kindAER = 'one'                     # one | all
mod_kindRF = 'one'                      # one | two | all
mod_kindGE = ''                         # '' | PUP
mod_biomeSHR = 'w/GRA'                  # SHR | w/GRA | w/FOR
mod_biomeURB = 'w/DES'                  # URB | w/DES

data_EFF = 'CDIAC'                      # CDIAC | EDGAR
data_LULCC = 'LUH1'                     # LUH1
data_ECH4 = 'EDGAR'                     # EDGAR | ACCMIP | EPA
data_EN2O = 'EDGAR'                     # EDGAR | EPA
data_Ehalo = 'EDGAR'                    # EDGAR
data_ENOX = 'EDGAR'                     # EDGAR | ACCMIP
data_ECO = 'EDGAR'                      # EDGAR | ACCMIP
data_EVOC = 'EDGAR'                     # EDGAR | ACCMIP
data_ESO2 = 'EDGAR'                     # EDGAR | ACCMIP
data_ENH3 = 'EDGAR'                     # EDGAR | ACCMIP
data_EOC = 'ACCMIP'                     # ACCMIP
data_EBC = 'ACCMIP'                     # ACCMIP
data_RFant = 'IPCC-AR5'                 # '' | IPCC-AR5
data_RFnat = 'IPCC-AR5'                 # '' | IPCC-AR5

mod_DATAscen = 'trends'                 # raw | offset | smoothX (X in yr) | trends

scen_ALL = 'SSP8.5'                           # '' | stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6

scen_EFF = 'SSP8.5'                       # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_LULCC = 'stop'                     # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ECH4 = 'SSP8.5'                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_EN2O = 'SSP8.5'                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_Ehalo = 'stop'                     # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ENOX = 'SSP8.5'                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ECO = 'SSP8.5'                       # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_EVOC = 'SSP8.5'                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ESO2 = 'SSP8.5'                      # stop | cst | SRES-A1B | SRES-A1FI | SRES-A1T | SRES-A2 | SRES-B1 | SRES-B2 | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_ENH3 = 'SSP8.5'                      # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_EOC = 'SSP8.5'                       # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_EBC = 'SSP8.5'                       # stop | cst | RCP8.5 | RCP6.0 | RCP4.5 | RCP2.6
scen_RFant = 'stop'                     # stop | cst
scen_RFnat = 'stop'                     # stop | cst

mod_OSNKstruct = 'HILDA'                # HILDA | BD-model | 2D-model | 3D-model
mod_OSNKchem = 'CO2SysPower'            # CO2SysPade | CO2SysPower
mod_OSNKtrans = 'mean-CMIP5'            # mean-CMIP5 | CESM1-BGC | IPSL-CM5A-LR | MPI-ESM-LR

mod_LSNKnpp = 'log'                     # log | hyp
mod_LSNKrho = 'exp'                     # exp | gauss
mod_LSNKpreind = 'mean-TRENDYv2'        # mean-TRENDYv2 | CLM-45 | JSBACH | JULES | LPJ | LPJ-GUESS | LPX-Bern | OCN | ORCHIDEE | VISIT
mod_LSNKtrans = 'mean-CMIP5'            # mean-CMIP5 | BCC-CSM-11 | CESM1-BGC | CanESM2 | HadGEM2-ES | IPSL-CM5A-LR | MPI-ESM-LR | NorESM1-ME
mod_LSNKcover = 'mean-TRENDYv2'         # ESA-CCI | MODIS | Ramankutty1999 | Levavasseur2012 | mean-TRENDYv2 | CLM-45 | JSBACH | JULES | LPJ | LPJ-GUESS | LPX-Bern | OCN | ORCHIDEE | VISIT

mod_EFIREpreind = 'mean-TRENDYv2'       # '' | mean-TRENDYv2 | CLM-45 | JSBACH | LPJ | LPJ-GUESS | ORCHIDEE | VISIT
mod_EFIREtrans = 'mean-CMIP5'           # '' | mean-CMIP5 | CESM1-BGC | IPSL-CM5A-LR | MPI-ESM-LR | NorESM1-ME

mod_ELUCagb = 'mean-TRENDYv2'           # mean-TRENDYv2 | CLM-45 | LPJ-GUESS | ORCHIDEE
mod_EHWPbb = 'high'                     # high | low
mod_EHWPtau = 'Earles2012'              # Houghton2001 | Earles2012
mod_EHWPfct = 'gamma'                   # gamma | lin | exp

mod_OHSNKtau = 'Prather2012'            # Prather2012 | CESM-CAM-superfast | CICERO-OsloCTM2 | CMAM | EMAC | GEOSCCM | GDFL-AM3 | GISS-E2-R | GISS-E2-R-TOMAS | HadGEM2 | LMDzORINCA | MIROC-CHEM | MOCAGE | NCAR-CAM-35 | STOC-HadAM3 | TM5 | UM-CAM
mod_OHSNKfct = 'lin'                    # lin | log
mod_OHSNKtrans = 'Holmes2013'           # mean-OxComp | Holmes2013 | GEOS-Chem | Oslo-CTM3 | UCI-CTM

mod_EWETpreind = 'mean-WETCHIMP'        # '' | mean-WETCHIMP | CLM-4Me | DLEM | IAP-RAS | LPJ-Bern | LPJ-WSL | ORCHIDEE | SDGVM
mod_AWETtrans = 'mean-WETCHIMP'         # '' | mean-WETCHIMP | CLM-4Me | DLEM | LPJ-Bern | ORCHIDEE | SDGVM | UVic-ESCM

mod_HVSNKtau = 'Prather2015'            # Prather2015 | GMI | GEOSCCM | G2d-M | G2d | Oslo-c29 | Oslo-c36 | UCI-c29 | UCI-c36
mod_HVSNKtrans = 'Prather2015'          # Prather2012 | Prather2015 | G2d | Oslo-c29 | UCI-c29
mod_HVSNKcirc = 'mean-CCMVal2'          # mean-CCMVal2 | AMTRAC | CAM-35 | CMAM | Niwa-SOCOL | SOCOL | ULAQ | UMUKCA-UCAM

mod_O3Tregsat = 'mean-HTAP'             # '' | mean-HTAP | CAMCHEM | FRSGCUCI | GISS-modelE | GMI | INCA | LLNL-IMPACT | MOZART-GFDL | MOZECH | STOC-HadAM3 | TM5-JRC | UM-CAM
mod_O3Temis = 'mean-ACCMIP'             # mean-OxComp | mean-ACCMIP | CICERO-OsloCTM2 | NCAR-CAM-35 | STOC-HadAM3 | UM-CAM
mod_O3Tclim = 'mean-ACCMIP'             # '' | mean-ACCMIP | CESM-CAM-superfast | GFDL-AM3 | GISS-E2-R | MIROC-CHEM | MOCAGE | NCAR-CAM-35 | STOC-HadAM3 | UM-CAM
mod_O3Tradeff = 'IPCC-AR5'              # IPCC-AR5 | IPCC-AR4 | mean-ACCMIP | CESM-CAM-superfast | CICERO-OsloCTM2 | CMAM | EMAC | GEOSCCM | GFDL-AM3 | GISS-E2-R | HadGEM2 | LMDzORINCA | MIROC-CHEM | MOCAGE | NCAR-CAM-35 | STOC-HadAM3 | UM-CAM | TM5

mod_O3Sfracrel = 'Newman2006'           # Newman2006 | Laube2013-HL | Laube2013-ML
mod_O3Strans = 'mean-CCMVal2'           # mean-CCMVal2 | AMTRAC | CCSR-NIES | CMAM | CNRM-ACM | LMDZrepro | MRI | Niwa-SOCOL | SOCOL | ULAQ | UMSLIMCAT | UMUKCA-UCAM
mod_O3Snitrous = 'Daniel2010-sat'       # '' | Daniel2010-sat | Daniel2010-lin
mod_O3Sradeff = 'IPCC-AR4'              # IPCC-AR4 | mean-ACCENT | ULAQ | DLR-E39C | NCAR-MACCM | CHASER

mod_SO4regsat = 'mean-HTAP'             # '' | mean-HTAP | CAMCHEM | GISS-PUCCINI | GMI | GOCART | INCA2 | LLNL-IMPACT | SPRINTARS
mod_SO4load = 'mean-ACCMIP'             # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_SO4radeff = 'mean-AeroCom2'         # mean-AeroCom2 | BCC | CAM4-Oslo | CAM-51 | GEOS-CHEM | GISS-MATRIX | GISS-modelE | GMI | GOCART | HadGEM2 | IMPACT-Umich | INCA | MPIHAM | NCAR-CAM-35 | OsloCTM2 | SPRINTARS

mod_POAconv = 'default'                 # default | GFDL | CSIRO
mod_POAregsat = 'mean-HTAP'             # '' | mean-HTAP | CAMCHEM | GISS-PUCCINI | GMI | GOCART | INCA2 | LLNL-IMPACT | SPRINTARS
mod_POAload = 'mean-ACCMIP'             # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_POAradeff = 'mean-AeroCom2'         # mean-AeroCom2 | BCC | CAM4-Oslo | CAM-51 | GEOS-CHEM | GISS-MATRIX | GISS-modelE | GMI | GOCART | HadGEM2 | IMPACT-Umich | INCA | MPIHAM | NCAR-CAM-35 | OsloCTM2 | SPRINTARS

mod_BCregsat = 'mean-HTAP'              # '' | mean-HTAP | CAMCHEM | GISS-PUCCINI | GMI | GOCART | INCA2 | LLNL-IMPACT | SPRINTARS
mod_BCload = 'mean-ACCMIP'              # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_BCradeff = 'mean-AeroCom2'          # mean-AeroCom2 | BCC | CAM4-Oslo | CAM-51 | GEOS-CHEM | GISS-MATRIX | GISS-modelE | GMI | GOCART | HadGEM2 | IMPACT-Umich | INCA | MPIHAM | NCAR-CAM-35 | OsloCTM2 | SPRINTARS
mod_BCadjust = 'Boucher2013'            # Boucher2013 | CSIRO | GISS | HadGEM2 | ECHAM5 | ECMWF

mod_NO3load = 'Bellouin2011'            # Bellouin2011 | Hauglustaine2014
mod_NO3radeff = 'mean-AeroCom2'         # mean-AeroCom2 | GEOS-CHEM | GISS-MATRIX | GMI | HadGEM2 | IMPACT-Umich | INCA | NCAR-CAM-35 | OsloCTM2

mod_SOAload = 'mean-ACCMIP'             # '' | mean-ACCMIP | GFDL-AM3 | GISS-E2-R
mod_SOAradeff = 'mean-AeroCom2'         # mean-oCom2 | CAM-51 | GEOS-CHEM | IMPACT-Umich | MPIHAM | OsloCTM2

mod_DUSTload = 'mean-ACCMIP'            # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_DUSTradeff = ''                     # [no value for now]

mod_SALTload = 'mean-ACCMIP'            # mean-ACCMIP | GFDL-AM3 | GISS-E2-R | MIROC-CHEM
mod_SALTradeff = ''                     # [no value for now]

mod_CLOUDsolub = 'Lamarque2011'         # Hansen2005 | Lamarque2011
mod_CLOUDerf = 'mean-ACCMIP'            # mean-ACCMIP | CSIRO-Mk360 | GFDL-AM3 | GISS-E2-R | HadGEM2 | LMDzORINCA | MIROC-CHEM | NCAR-CAM-51
mod_CLOUDpreind = 'median'              # low | median | high

mod_ALBBCreg = 'Reddy2007'              # Reddy2007
mod_ALBBCrf = 'mean-ACCMIP'             # mean-ACCMIP | CICERO-OsloCTM2 | GFDL-AM3 | GISS-E2-R | GISS-E2-R-TOMAS | HadGEM2 | MIROC-CHEM | NCAR-CAM-35 | NCAR-CAM-51
mod_ALBBCwarm = 'median'                # low | median | high 

mod_ALBLCflux = 'CERES'                 # CERES | GEWEX | MERRA
mod_ALBLCalb = 'GlobAlbedo'             # GlobAlbedo | MODIS
mod_ALBLCcover = 'ESA-CCI'              # ESA-CCI | MODIS
mod_ALBLCwarm = 'Jones2013'             # Hansen2005 | Davin2007 | Davin2010 | Jones2013

mod_TEMPresp = 'mean-CMIP5'             # mean-CMIP5 | ACCESS-10 | ACCESS-13 | BCC-CSM-11 | BCC-CSM-11m | CanESM2 | CCSM4 | CNRM-CM5 | CNRM-CM5-2 | CSIRO-Mk360 | GFDL-CM3 | GFDL-ESM2G | GFDL-ESM2M | GISS-E2-H | GISS-E2-R | HadGEM2-ES | IPSL-CM5A-LR | IPSL-CM5A-MR | IPSL-CM5B-LR | MIROC5 | MIROC-ESM | MPI-ESM-LR | MPI-ESM-MR | MPI-ESM-P | MRI-CGCM3 | NorESM1-M
mod_TEMPpattern = 'hist&RCPs'           # 4xCO2 | hist&RCPs

mod_PRECresp = 'mean-CMIP5'             # mean-CMIP5 | ACCESS-10 | ACCESS-13 | BCC-CSM-11 | BCC-CSM-11m | CanESM2 | CCSM4 | CNRM-CM5 | CNRM-CM5-2 | CSIRO-Mk360 | GFDL-CM3 | GFDL-ESM2G | GFDL-ESM2M | GISS-E2-H | GISS-E2-R | HadGEM2-ES | IPSL-CM5A-LR | IPSL-CM5A-MR | IPSL-CM5B-LR | MIROC5 | MIROC-ESM | MPI-ESM-LR | MPI-ESM-MR | MPI-ESM-P | MRI-CGCM3 | NorESM1-M
mod_PRECradfact = 'Andrews2010'         # Andrews2010 | Kvalevag2013
mod_PRECpattern = 'hist&RCPs'           # 4xCO2 | hist&RCPs

mod_ACIDsurf = 'Bernie2010'             # Tans2009 | Bernie2010

mod_SLR = ''                            # [no value for now]
## Choose input files
Filein = 'UpAreaYieldT'


##choose Yc-Tc types
TYform = 'Qua'  # Qua / Log / Log2.5
ty=0
# Quadratic function
# For quadratic equation: 1. central case wheat
ParGlo_A = [-0.0065]; ParGlo_B = [0.117]; ParGlo_C = [2.0729] # Global, 3,6,7,9
ParNAm_A = [-0.0072]; ParNAm_B = [0.143]; ParNAm_C = [1.8667] # USA for North America,1
ParSEA_A = [-0.0064]; ParSEA_B = [0.1152]; ParSEA_C = [2.0416] # India for South& Southeast Aisa,8
ParAfr_A = [-0.0073]; ParAfr_B = [0.1314]; ParAfr_C = [2.3287] # Sudan for Africa,4 & 5
ParSAm_A = [-0.0057]; ParSAm_B = [0.1026]; ParSAm_C = [1.8183] # Mexico for South America,2

# Summarize to Para_A, Para_B and Para_C
Para_A = [0,-0.0072,-0.0057,-0.0065,-0.0073,-0.0073,-0.0065,-0.0065,-0.0064,-0.0065]
Para_B = [0,0.143, 0.1026, 0.117, 0.1314, 0.1314, 0.117, 0.117, 0.1152, 0.117]
Para_C = [0,1.8667,1.8183,2.0729,2.3287,2.3287, 2.0729, 2.0729, 2.0416, 2.0729]

# Maize
MaiGlo_A = [-0.0104]; MaiGlo_B = [0.3952]; MaiGlo_C = [-2.2224] # Global, 1, 2,,3, 6, 9
MaiCHN_A = [-0.0146]; MaiCHN_B = [0.5256]; MaiCHN_C = [-3.0037] # China for China region, 7
MaiAfr_A = [-0.0074]; MaiAfr_B = [0.2634]; MaiAfr_C = [-0.7861] # Africa, 4 & 5
MaiAfr_A = [-0.0152]; MaiAfr_B = [0.5776]; MaiAfr_C = [-3.8926] # Pakistan for South& Southeast Aisa,8

# Summarize to Mai_A, Mai_B and Mai_C
Mai_A = [0,-0.0104,-0.0104,-0.0104,-0.0074,-0.0074,-0.0104,-0.0146,-0.0152,-0.0104];
Mai_B = [0,0.3952,0.3952,0.3952,0.2634,0.2634,0.3952,0.5256,0.5776,0.3952];
Mai_C = [0,-2.2224,-2.2224,-2.2224,-0.7861,-0.7861,-2.2224,-3.0037,-3.8926,-2.2224];

# ratio of biomass transport and treatment
RatTT = 0.092


# Log+Qua function
#tyLN = 0
Para_S = [-0.0058]
Para_I = [0.10932]

# Log+2.5 function function
Para_1 = [-0.0015]
Para_2 = [0.1195]

##choose Yc-CO2 types
Tc = 0
TCO2_A = [-0.000002382,-0.0000011,-0.00000653,-0.0000074]
TCO2_B = [0.003335,0.001184,0.010735,0.012715]
TCO2_C = [0.124615,4.2314,4.5291,2.9514]

## Starting year of BECCS
Ybeccs = 2050                        ## 2030,2040,2050,2060,2070,2080,2090,2100

## area/capita type
AreaType = '25Ad2Smo'                            ## '05Ad29Smo','05Ad27Smo','05Ad23Smo'
## FROM Year Ys to predict npp and yield
Ys = 2030

## From year YsAr to predict area
YsAr = 2025

ExpSeq = 'Mar1For2'              # 'For1Mar2', 'Mar1For2'

## Open or close GHG emission, 1-EGHG=0; 0-EGHG not = 0
EGHG = 0
## Cm Cn, N2O concentration D_N2O
Cm = 2; Cn = 0
## N2O function, Pf - peak value; Tf - final time to reach peak value
Pf = 150; Tf = 350
### CliSen -- lambda_0
#CliSen = 1.4

##################################################
#   2. OSCAR
##################################################

if (scen_ALL != ''):
    scen_LULCC = scen_Ehalo = scen_RFant = scen_RFnat = scen_ALL
    if (scen_RFant[:3] == 'SSP'):
        scen_RFant = 'cst'
    if (scen_RFnat[:3] == 'SSP'):
        scen_RFnat = 'cst'

execfile('OSCAR-loadD.py')
execfile('OSCAR-loadP.py')
execfile('OSCAR-format.py')
execfile('OSCAR-fct.py')
#execfile('OSCAR-fct-StopAgriClim.py') # close agriculture affect climate, close climate affect agriculture
#execfile('OSCAR-fct-OpenAgriClim.py') # open agriculture affect climate, open climate affect agriculture
#execfile('OSCAR-fct-OpenAgriStopClim.py') # open agriculture affect climate, close climate affect agriculture
#execfile('OSCAR-fct-StopAgriOpenClim.py') # stop agriculture affect climate, open climate affect agriculture


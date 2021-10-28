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


import csv

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties


##################################################
#   1. OSCAR LITE
##################################################

def OSCAR_lite(p=p,fT=fT,\
               EFF=EFF,ECH4=ECH4,EN2O=EN2O,\
               LUC=LUC,HARV=HARV,SHIFT=SHIFT,\
               EHFC=EHFC,EPFC=EPFC,EODS=EODS,\
               ENOX=ENOX,ECO=ECO,EVOC=EVOC,ESO2=ESO2,ENH3=ENH3,EOC=EOC,EBC=EBC,\
               RFcon=RFcon,RFvolc=RFvolc,RFsolar=RFsolar,\
               force_CO2=False,force_GHG=False,force_halo=False,force_RF=False,force_RFs=False,force_clim=False,\
               var_output=['ELUC','OSNK','LSNK','D_CO2','RF','D_gst','D_EBB_CO2','D_EBB_CH4','D_EBB_N2O','BECCSall','NPP_crop','ProdGlobalCap','EnerDCGlo','ProdDCGloCap','YieldTot',\
                           'D_EBB_NOX','D_EBB_CO','D_EBB_VOC','D_EBB_SO2','D_EBB_NH3','D_EBB_OC','D_EBB_BC','NPP','Yieldwh_cou','Yieldri_cou','Yieldma_cou','Yieldso_cou','Yieldot_cou','Areawh_cou','Areari_cou','Areama_cou','Areaso_cou','Areaot_cou'],\
               plot=[]):

    #===============
    # A. DEFINITIONS
    #===============

    # plot variables
    var_plot = []
    if plot is 'all' or plot is 'CO2' or 'CO2' in plot:
        var_plot += ['D_CO2','OSNK','LSNK','ELUC','D_AREA','D_npp','D_efire','D_fmort','D_rh1','D_fmet','D_rh2','Diff','D_FIN','D_FOUT','D_FCIRC','EFIRE_luc','FMORT_luc','RH1_luc','FMET_luc','RH2_luc','EHWP1_luc','EHWP2_luc','EHWP3_luc','bLSNK','b1ELUC','b2ELUC','b1RH1_luc','b2RH1_luc','b1RH2_luc','b2RH2_luc','b1EFIRE_luc','b2EFIRE_luc','b1EHWP1_luc','b2EHWP1_luc','b1EHWP2_luc','b2EHWP2_luc','b1EHWP3_luc','b2EHWP3_luc']
    if plot is 'all' or plot is 'CH4' or 'CH4' in plot:
        var_plot += ['D_CH4','D_OHSNK_CH4','D_HVSNK_CH4','D_XSNK_CH4','D_EWET','D_EBB_CH4']
    if plot is 'all' or plot is 'N2O' or 'N2O' in plot:
        var_plot += ['D_N2O','D_HVSNK_N2O','D_EBB_N2O']
    if plot is 'all' or plot is 'O3' or 'O3' in plot:
        var_plot += ['D_O3t','D_O3s','D_EESC','D_N2O_lag','D_gst']
    if plot is 'all' or plot is 'AER' or 'AER' in plot:
        var_plot += ['D_SO4','D_POA','D_BC','D_NO3','D_SOA','D_AERh','RF_SO4','RF_POA','RF_BC','RF_NO3','RF_SOA','RF_cloud']
    if plot is 'all' or plot is 'clim' or 'clim' in plot:
        var_plot += ['RF','D_gst','D_gyp','RF_CO2','RF_CH4','RF_H2Os','RF_N2O','RF_halo','RF_O3t','RF_O3s','RF_SO4','RF_POA','RF_BC','RF_NO3','RF_SOA','RF_cloud','RF_BCsnow','RF_LCC']

    # save variables
    var_timeseries = list(set(var_output)|set(var_plot))
    for var in var_timeseries:
        # global variables
        if var in ['D_mld','D_dic','D_pH']\
        or var in ['OSNK','LSNK','D_OHSNK_CH4','D_HVSNK_CH4','D_XSNK_CH4','D_HVSNK_N2O','ProdGlobalCap','EnerDCGlo','ProdDCGloCap','BECCSall','LSNKfo','LSNKcr4','YieldTot']\
        or var in ['D_kOH','D_hv']\
        or var in ['D_O3t','D_EESC','D_O3s','D_SO4','D_POA','D_BC','D_NO3','D_SOA','D_AERh']\
        or var in ['D_CO2','D_CH4','D_CH4_lag','D_N2O','D_N2O_lag']\
        or var in ['RF','RF_warm','RF_atm','RF_CO2','RF_CH4','RF_H2Os','RF_N2O','RF_halo','RF_O3t','RF_O3s','RF_SO4','RF_POA','RF_BC','RF_NO3','RF_SOA','RF_cloud','RF_BCsnow','RF_LCC']\
        or var in ['D_gst','D_sst','D_gyp','D_OHC']:
            exec(var+'_t = np.zeros([ind_final+1],dtype=dty)')
        # (region) variables
        if var in ['ELUC','D_AWET','D_EWET','D_ewet','D_EBB_CO2','D_EBB_CH4','D_EBB_N2O','D_EBB_NOX','D_EBB_CO','D_EBB_VOC','D_EBB_SO2','D_EBB_NH3','D_EBB_OC','D_EBB_BC','D_lst','D_lyp','NPP_crop']:
            exec(var+'_t = np.zeros([ind_final+1,nb_regionI],dtype=dty)')        
        # (region)*(biome) variables
        if var in ['D_AREA','D_npp','D_efire','D_fmort','D_rh1','D_fmet','D_rh2','Diff','D_cveg','D_csoil1','D_csoil2','NPP','bLSNK','b1ELUC','b2ELUC','b1RH1_luc','b2RH1_luc','b1RH2_luc','b2RH2_luc','b1EFIRE_luc','b2EFIRE_luc','b1EHWP1_luc','b2EHWP1_luc','b1EHWP2_luc','b2EHWP2_luc','b1EHWP3_luc','b2EHWP3_luc']:
            exec(var+'_t = np.zeros([ind_final+1,nb_regionI,nb_biome],dtype=dty)')
        # (region)*(biome)*(biome)*(age) variables
        if var in ['EFIRE_luc','FMORT_luc','RH1_luc','FMET_luc','RH2_luc','EHWP1_luc','EHWP2_luc','EHWP3_luc']\
        or var in ['CVEG_luc','CSOIL1_luc','CSOIL2_luc','CHWP1_luc','CHWP2_luc','CHWP3_luc']:
            exec(var+'_t = np.zeros([ind_final+1,nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty)')        
        # (obox) variables
        if var in ['D_FIN','D_FOUT','D_FCIRC','D_CSURF']:
            exec(var+'_t = np.zeros([ind_final+1,nb_obox],dtype=dty)')
        # (species) variables
        if var in ['D_HFC','D_HFC_lag','D_OHSNK_HFC','D_HVSNK_HFC','D_XSNK_HFC']:
            exec(var+'_t = np.zeros([ind_final+1,nb_HFC],dtype=dty)')
        if var in ['D_PFC','D_PFC_lag','D_OHSNK_PFC','D_HVSNK_PFC','D_XSNK_PFC']:
            exec(var+'_t = np.zeros([ind_final+1,nb_PFC],dtype=dty)')
        if var in ['D_ODS','D_ODS_lag','D_OHSNK_ODS','D_HVSNK_ODS','D_XSNK_ODS']:
            exec(var+'_t = np.zeros([ind_final+1,nb_ODS],dtype=dty)')
        # (country) variables
        if var in ['Yieldwh_cou','Yieldri_cou','Yieldma_cou','Yieldso_cou','Yieldot_cou','Areawh_cou','Areari_cou','Areama_cou','Areaso_cou','Areaot_cou','Prodwh_cou','Prodri_cou','Prodma_cou','Prodso_cou','Prodot_cou','ProdDC_cou']:
            exec(var+'_t = np.zeros([ind_final+1,167],dtype=dty)') # 167 represent 167 countries

    # run variables
    # ocean
    D_dic = np.array([0],dtype=dty)
    D_CSURF = np.zeros([nb_obox],dtype=dty)
    # land
    for var in ['D_AREA','D_cveg','D_csoil1','D_csoil2']:
        exec(var+' = np.zeros([nb_regionI,nb_biome],dtype=dty)')
    # land-use
    for var in ['CVEG_luc','CSOIL1_luc','CSOIL2_luc','CHWP1_luc','CHWP2_luc','CHWP3_luc']:
        exec(var+' = np.zeros([nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty)')
    # atmosphere
    for var in ['D_CO2','D_CH4','D_CH4_lag','D_N2O','D_N2O_lag','D_EESC','D_O3s']:
        exec(var+' = np.array([0],dtype=dty)')
    for var in ['D_HFC','D_HFC_lag','D_PFC','D_PFC_lag','D_ODS','D_ODS_lag']:
        exec(var+' = np.zeros([nb_'+var[2:2+3]+'],dtype=dty)')
    # climate
    for var in ['D_gst','D_gst0','D_sst','D_gyp','D_OHC']:
        exec(var+' = np.array([0],dtype=dty)')
    for var in ['D_lst','D_lyp']:
        exec(var+' = np.zeros([nb_regionI],dtype=dty)')

    
    
    #=======
    # B. RUN
    #=======
    #get N input from FAO and fitted
    
    newbiome = ['wh','ri','ma','so']
    #Sensitivity of Nitrogen to npp
    CouID = np.array([line for line in csv.reader(open('data/'+Filein+'/AreaCoun167.csv','r'))], dtype=dty)[:,0]
    #Define
    D_CO2_2018 = np.zeros([1,1],dtype=dty)
    D_lst_2018 = np.zeros([1,10],dtype=dty)
    D_lyp_2018 = np.zeros([1,10],dtype=dty)
    AlphaRest = np.zeros([10],dtype=dty)
    AlphaW = np.zeros([10],dtype=dty)
    AlphaR = np.zeros([10],dtype=dty)
    AlphaM = np.zeros([10],dtype=dty)
    AlphaS = np.zeros([10],dtype=dty)
    RatYie = np.zeros([10],dtype=dty)
    
    Yieldwh_cou = np.zeros(167) #'Yieldwh_cou','Yieldri_cou','Yieldma_cou','Yieldso_cou'
    Yieldri_cou = np.zeros(167) #'Yieldwh_cou','Yieldri_cou','Yieldma_cou','Yieldso_cou'
    Yieldma_cou = np.zeros(167) #'Yieldwh_cou','Yieldri_cou','Yieldma_cou','Yieldso_cou'
    Yieldso_cou = np.zeros(167) #'Yieldwh_cou','Yieldri_cou','Yieldma_cou','Yieldso_cou'
    Yieldot_cou = np.zeros(167)
    
    Areawh_cou = np.zeros(167)
    Areari_cou = np.zeros(167)
    Areama_cou = np.zeros(167)
    Areaso_cou = np.zeros(167)
    Areaot_cou = np.zeros(167)
    
    ProdGlobalCap = np.zeros([1])
    EnerDCGlo = np.zeros([1])
    ProdDCGloCap = np.zeros([1])
    YieldTot = np.zeros([1])
    
    ## npp in FAO,Absolute value
    nppFAO = np.array([line for line in csv.reader(open('data/'+Filein+'/nppAbFAO1961to2030.csv','r'))], dtype=dty)
    ### npp need to multiply 0.2
    nppFAO *= 0.2
    nppFAOWh = nppFAO[:,0:10]
    nppFAORi = nppFAO[:,10:20]
    nppFAOMa = nppFAO[:,20:30]
    nppFAOSo = nppFAO[:,30:40]
    nppFAOOt = nppFAO[:,40:50]
    
    ## fBECCS
    Fbe = np.array([line for line in csv.reader(open('data/'+Filein+'/fbeccs.csv','r'))], dtype=dty)
    
    ## Population
    Pop = np.array([line for line in csv.reader(open('data/'+Filein+'/PopulationFAO.csv','r'))], dtype=dty)
    
    ## To calculate crop energy
    WeiOth = np.array([line for line in csv.reader(open('data/'+Filein+'/OthersData.csv','r'))], dtype=dty)
    ConEner = np.array([line for line in csv.reader(open('data/'+Filein+'/EnerConvUp.csv','r'))], dtype=dty)
    ## Calculate carolie factors provided by food for countries and crops, * ratio of calorie * ratio of food
    Fcp = np.zeros([167,5])
    
    for c in range(167):
        Fcp[c,0:4] = ConEner[0:4].reshape(4)
        Fcp[c,4] = np.sum(WeiOth[c,:][:,np.newaxis]*ConEner[4:])
    
    ## Countries area 
    AreaPCou = np.array([line for line in csv.reader(open('data/'+Filein+'/AreaCounF'+str(YsAr)+'_'+AreaType+'2500.csv','r'))], dtype=dty)
    fy = int(AreaPCou[0,0])
    AreaPCouWh = AreaPCou[:,1:168]
    AreaPCouRi = AreaPCou[:,169:336]
    AreaPCouMa = AreaPCou[:,337:504]
    AreaPCouSo = AreaPCou[:,505:672]
    AreaPCouOt = AreaPCou[:,673:840]
    
    ## Temperature at 1700
    T0 = np.array([0, 18.87 , 22.15 , 18.06 , 22.62 , 23.28 , 16.92 , 20.72 , 25.00 , 21.17])
    
    ## Developed countries, 1- yes, 2- no
    CodeDC = np.array([line for line in csv.reader(open('data/'+Filein+'/CodeDevCou.csv','r'))], dtype=dty)
    ## Tropical region,1-yes,0-no  
    CodeTR = np.array([line for line in csv.reader(open('data/'+Filein+'/CodeTropic235.csv','r'))], dtype=dty)
    
    if Ybeccs < 2500:
        TMP = np.array([line for line in csv.reader(open('data/'+Filein+'/EFFpath_RCP'+scen_EFF[3]+scen_EFF[5]+'.csv','r'))], dtype=dty) # 2000-2500
        x = (Ybeccs-2030)/10
        EFFpath = TMP[:,x]
        for n in range(Ybeccs-1700,ind_final+1):
            EFF[n,:] = EFFpath[n-300]/10
        
    ## Calculate additional emissions from expansion for increased nitrogen input
    # conversion, 298*12/44*10^(-6)
    Emi_Soil = 4890 # kt N2O
    Emi_exp = np.zeros([801,1])
    AreaTotal = np.sum(AreaPCouWh,1)+np.sum(AreaPCouRi,1)+np.sum(AreaPCouMa,1)+np.sum(AreaPCouSo,1)+np.sum(AreaPCouOt,1)
    for t in range(YsAr-1701,801):
        Emi_exp[t,0] = Emi_Soil*0.33*298*12/44*10**(-6)*(AreaTotal[t-319]/AreaTotal[0])
               
    # area change of non-cereal and cereals
    for t in range(1,ind_final+1):
        
        if t>50 and t< 317 :
            AreaLUCRest = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCRest_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaLUCW = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCWheat_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaLUCR = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCRice_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaLUCM = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCMaize_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaLUCS = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCSoybeans_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaOri = (np.sum(LUC[t],1) - np.sum(LUC[t],2))
            # calculate the ratio
            AlphaRest[1:10] = AreaLUCRest[t]/AreaOri[1:10,3]
            AlphaW[1:10] = AreaLUCW[t]/AreaOri[1:10,3]
            AlphaR[1:10] = AreaLUCR[t]/AreaOri[1:10,3]
            AlphaM[1:10] = AreaLUCM[t]/AreaOri[1:10,3]
            AlphaS[1:10] = AreaLUCS[t]/AreaOri[1:10,3]
            # change respective LUC by the ratio
            LUC[t,:,5,:] = AlphaW[:,np.newaxis]*LUC[t,:,3,:] 
            LUC[t,:,:,5] = AlphaW[:,np.newaxis]*LUC[t,:,:,3] 
            LUC[t,:,6,:] = AlphaR[:,np.newaxis]*LUC[t,:,3,:] 
            LUC[t,:,:,6] = AlphaR[:,np.newaxis]*LUC[t,:,:,3] 
            LUC[t,:,7,:] = AlphaM[:,np.newaxis]*LUC[t,:,3,:] 
            LUC[t,:,:,7] = AlphaM[:,np.newaxis]*LUC[t,:,:,3] 
            LUC[t,:,8,:] = AlphaS[:,np.newaxis]*LUC[t,:,3,:] 
            LUC[t,:,:,8] = AlphaS[:,np.newaxis]*LUC[t,:,:,3] 
            LUC[t,:,3,:] = AlphaRest[:,np.newaxis]*LUC[t,:,3,:] 
            LUC[t,:,:,3] = AlphaRest[:,np.newaxis]*LUC[t,:,:,3] 
            # change respective SHIFT by the ratio of SHIFT
            SHIFT[t,:,5,:] = AlphaW[:,np.newaxis]*SHIFT[t,:,3,:] 
            SHIFT[t,:,:,5] = AlphaW[:,np.newaxis]*SHIFT[t,:,:,3] 
            SHIFT[t,:,6,:] = AlphaR[:,np.newaxis]*SHIFT[t,:,3,:] 
            SHIFT[t,:,:,6] = AlphaR[:,np.newaxis]*SHIFT[t,:,:,3] 
            SHIFT[t,:,7,:] = AlphaM[:,np.newaxis]*SHIFT[t,:,3,:] 
            SHIFT[t,:,:,7] = AlphaM[:,np.newaxis]*SHIFT[t,:,:,3] 
            SHIFT[t,:,8,:] = AlphaS[:,np.newaxis]*SHIFT[t,:,3,:]
            SHIFT[t,:,:,8] = AlphaS[:,np.newaxis]*SHIFT[t,:,:,3] 
            SHIFT[t,:,3,:] = AlphaRest[:,np.newaxis]*SHIFT[t,:,3,:] 
            SHIFT[t,:,:,3] = AlphaRest[:,np.newaxis]*SHIFT[t,:,:,3] # 

        elif t > 316:
            LUC[t,1:,1,:] = 0 
            AreaLUCRest = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCRest_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaLUCW = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCWheat_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaLUCR = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCRice_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaLUCM = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCMaize_'+AreaType+'2500.csv','r'))], dtype=dty)
            AreaLUCS = np.array([line for line in csv.reader(open('data/'+Filein+'/LUCSoybeans_'+AreaType+'2500.csv','r'))], dtype=dty)
            ## get rid of other disturbation
            LUC[t,1:,:,3] = 0
            LUC[t,1:,3,:] = 0  

            if t<325:
                LUC[t,1:,1,3] = AreaLUCRest[t,:]
                LUC[t,1:,1,5] = AreaLUCW[t,:]
                LUC[t,1:,1,6] = AreaLUCR[t,:]
                LUC[t,1:,1,7] = AreaLUCM[t,:]
                LUC[t,1:,1,8] = AreaLUCS[t,:]
            elif t>324:
                # 1. first marginal land, then forest
                if ExpSeq == 'Mar1For2':
                    m = 0.2; n=1-m
                    LUC[t,1:,0,3] = m*AreaLUCRest[t,:]
                    LUC[t,1:,0,5] = m*AreaLUCW[t,:]
                    LUC[t,1:,0,6] = m*AreaLUCR[t,:]
                    LUC[t,1:,0,7] = m*AreaLUCM[t,:]
                    LUC[t,1:,0,8] = m*AreaLUCS[t,:]
                    
                    LUC[t,1:,1,3] = n*AreaLUCRest[t,:]
                    LUC[t,1:,1,5] = n*AreaLUCW[t,:]
                    LUC[t,1:,1,6] = n*AreaLUCR[t,:]
                    LUC[t,1:,1,7] = n*AreaLUCM[t,:]
                    LUC[t,1:,1,8] = n*AreaLUCS[t,:]
                  
                # 2. first forest, then marginal land
                elif ExpSeq == 'For1Mar2':
                    LUC[t,1:,1,3] = AreaLUCRest[t,:]
                    LUC[t,1:,1,5] = AreaLUCW[t,:]
                    LUC[t,1:,1,6] = AreaLUCR[t,:]
                    LUC[t,1:,1,7] = AreaLUCM[t,:]
                    LUC[t,1:,1,8] = AreaLUCS[t,:]
                            
            
        for tt in range(p):

            #---------
            # 1. OCEAN
            #---------

            # structure
            D_mld = mld_0 * alpha_mld * (np.exp(gamma_mld*fT*D_sst)-1)
            # fluxes
            D_FIN = p_circ * v_fg * alpha_CO2 * D_CO2
            D_FOUT = p_circ * v_fg * alpha_CO2 * f_pCO2(D_dic,fT*D_sst)
            D_FCIRC = D_CSURF * (1/tau_circ)
            OSNK = np.sum(D_FOUT - D_FIN)
            # stocks
            D_CSURF += (p**-1) * (D_FIN - D_FOUT - D_FCIRC)
#            D_dic = alpha_dic * np.sum(D_CSURF) / (1+D_mld/mld_0)
            D_dic = max(alpha_dic * np.sum(D_CSURF) / (1+D_mld/mld_0),-1*dic_0)
            #--------
            # 2. LAND
            #--------
            # land-cover
            ### read yield at 2017 for 167 countries *
            YieldwhYs = np.array([line for line in csv.reader(open('data/'+Filein+'/Yield5_Coun167_'+str(Ys)+'.csv','r'))], dtype=dty)[:,1]
            YieldriYs = np.array([line for line in csv.reader(open('data/'+Filein+'/Yield5_Coun167_'+str(Ys)+'.csv','r'))], dtype=dty)[:,2]
            YieldmaYs = np.array([line for line in csv.reader(open('data/'+Filein+'/Yield5_Coun167_'+str(Ys)+'.csv','r'))], dtype=dty)[:,3]
            YieldsoYs = np.array([line for line in csv.reader(open('data/'+Filein+'/Yield5_Coun167_'+str(Ys)+'.csv','r'))], dtype=dty)[:,4]
            YieldotYs = np.array([line for line in csv.reader(open('data/'+Filein+'/Yield5_Coun167_'+str(Ys)+'.csv','r'))], dtype=dty)[:,5]
    
            D_lyp_cou = np.zeros([167])
            D_lyp_Y = np.zeros([167])
            D_lyp_couB = np.zeros([167])
            D_CO2_B = np.zeros([1])
            FT = np.zeros([167]); FT_Ma = np.zeros([167])
            FP = np.zeros([167])
            
            FTReg  = np.zeros([10]); FTt  = np.zeros([10]); FTs  = np.zeros([10])
            FTReg_Ma  = np.zeros([10]); FTt_Ma  = np.zeros([10]); FTs_Ma  = np.zeros([10])
            
            ## Correct T0 to make T2016-2019 is equal to observed
            if t == 320:
                Tmod = T0 + np.mean(np.array([D_lst_t[316,:],D_lst_t[317,:],D_lst_t[318,:],D_lst_t[319,:]]),axis=0)
                Tobs=np.array([0,21.56573923,24.2579048,20.19866791,24.99114654,25.52582281,19.85079192,22.94499806,26.94859918,23.12200498])
                Tcorr = Tobs-Tmod
                T0 = T0+Tcorr
                
            CouID = np.array([line for line in csv.reader(open('data/'+Filein+'/AreaCoun167.csv','r'))], dtype=dty)[:,0]
            if t==Ys-1700:
                Yieldwh_cou = YieldwhYs # cropland-cereals npp
                Yieldri_cou = YieldriYs # cropland-cereals npp
                Yieldma_cou = YieldmaYs # cropland-cereals npp
                Yieldso_cou = YieldsoYs # cropland-cereals npp
                Yieldot_cou = YieldotYs # cropland-cereals npp
            
            ## get temperature change and yield change relationship
            if t> Ys-1700:
                D_Ts = D_lst_t[Ys-1700,:]
                D_CO2_Y = D_CO2_t[Ys-1700] # CO2 at the year 2017
                D_lyp_YReg = D_lyp_t[Ys-1700,:] # precipitation at the year 2017
                 # FCO2 at first BECCS year 
                #FCO2, FCO2_Y, FP( FP_Y )
                FCY = (TCO2_A[Tc]*(D_CO2_Y+278)**2 + TCO2_B[Tc]*(D_CO2_Y+278)+ TCO2_C[Tc])/(TCO2_A[Tc]*278**2 + TCO2_B[Tc]*278+ TCO2_C[Tc])
                if D_CO2+278<700:
                    FC = (TCO2_A[Tc]*(D_CO2+278)**2 + TCO2_B[Tc]*(D_CO2+278)+ TCO2_C[Tc])/(TCO2_A[Tc]*278**2 + TCO2_B[Tc]*278+ TCO2_C[Tc])
                elif D_CO2+278>700 and D_CO2+278==700:
                    FC = (TCO2_A[Tc]*700**2 + TCO2_B[Tc]*700+ TCO2_C[Tc])/(TCO2_A[Tc]*278**2 + TCO2_B[Tc]*278+ TCO2_C[Tc])
                if D_CO2 > D_CO2_Y:
                    FC_Ma = FC
                elif D_CO2 <= D_CO2_Y:
                    FC_Ma = FCY
                
                #FT
                if TYform == 'Qua':
                    ## 1. original function
                    tmin = 0.01*(Para_C[ty] - (Para_B[ty]**2)/(4*Para_A[ty])) # 1% of Ymax,Ymax = c-b^2/4*a
                    tmin_Ma = Mai_A[0]*(30-8.5)**2 + Mai_B[0]*(30-8.5)+ Mai_C[0]
                    Xma = -Mai_B[0]/2/Mai_A[0]
                    for n in range(10):
                        FTt[n] = max(tmin,Para_A[ty]*(T0[n]+D_lst[n])**2 + Para_B[ty]*(T0[n]+D_lst[n]) + Para_C[ty])
                        FTs[n] = max(tmin,Para_A[ty]*(T0[n]+D_Ts[n])**2 + Para_B[ty]*(T0[n]+D_Ts[n]) + Para_C[ty])
                        FTReg[n] = FTt[n]/FTs[n]
                        if FTs[n] == 0:
                            FTReg[n] = 0
                        # FOR MAIZE
                        FTt_Ma[n] = max(tmin_Ma,Mai_A[0]*(T0[n]+D_lst[n])**2 + Mai_B[0]*(T0[n]+D_lst[n]) + Mai_C[0])
                        FTs_Ma[n] = max(tmin_Ma,Mai_A[0]*(T0[n]+D_Ts[n])**2 + Mai_B[0]*(T0[n]+D_Ts[n]) + Mai_C[0])
                        if T0[n]+D_lst[n] <Xma:
                            FTt_Ma[n] = Mai_A[0]*(Xma)**2 + Mai_B[0]*(Xma) + Mai_C[0]
                        if T0[n]+D_Ts[n] <Xma:
                            FTs_Ma[n] = Mai_A[0]*(Xma)**2 + Mai_B[0]*(Xma) + Mai_C[0]
                        
                        FTReg_Ma[n] = FTt_Ma[n]/FTs_Ma[n]
                        if FTs_Ma[n] == 0:
                            FTReg_Ma[n] = 0
                
                ## Calculate yield for each country
                for i in range(167):
                    CI = int(CouID[i])
                    D_lyp_Y[i] = D_lyp_YReg[CI]
                    D_lyp_cou[i] = D_lyp[CI]
                    FT[i] = FTReg[CI]; FT_Ma[i] = FTReg_Ma[CI]
                    ## Calculate FP
                    FP[i] = np.exp(gamma_yieldP[i]*D_lyp_cou[i])/np.exp(gamma_yieldP[i]*D_lyp_Y[i])
                    
                    Yieldwh_cou[i] = max(0,YieldwhYs[i] * f_yieldwhYStart(FC,FCY,FT,FP)[i]) 
                    Yieldri_cou[i] = max(0,YieldriYs[i] * f_yieldriYStart(FC,FCY,FT,FP)[i]) 
                    Yieldma_cou[i] = max(0,YieldmaYs[i] * f_yieldmaYStart(FC_Ma,FCY,FT_Ma,FP)[i]) 
                    Yieldso_cou[i] = max(0,YieldsoYs[i] * f_yieldsoYStart(FC,FCY,FT,FP)[i]) 
                    Yieldot_cou[i] = max(0,YieldotYs[i] * f_yieldotYStart(FC,FCY,FT,FP)[i]) 
                           
            if t > fy-1701:
                Areawh_cou = AreaPCouWh[t-fy+1700] 
                Areari_cou = AreaPCouRi[t-fy+1700] 
                Areama_cou = AreaPCouMa[t-fy+1700] 
                Areaso_cou = AreaPCouSo[t-fy+1700] 
                Areaot_cou = AreaPCouOt[t-fy+1700] 
                
                
            D_AREA += (p**-1) * (np.sum(LUC[t],1) - np.sum(LUC[t],2)) 
            
            D_AWET = AWET_0*(gamma_wetT*fT*D_lst + gamma_wetP*fT*D_lyp + gamma_wetC*fT*D_CO2)
            # factors
            D_k_igni = gamma_igniT*fT*D_lst[:,np.newaxis]+gamma_igniP*fT*D_lyp[:,np.newaxis]+gamma_igniC*fT*D_CO2
            D_k_rho = f_rho(fT*D_lst[:,np.newaxis],fT*D_lyp[:,np.newaxis])
            # fluxes
            D_npp = npp_0 * f_npp(D_CO2,fT*D_lst[:,np.newaxis],fT*D_lyp[:,np.newaxis])
                
            if t>261 and t < Ys-1699: # equal to FAO 1961-2020
                D_npp[:,3] = nppFAOOt[t-261,:] - npp_0[:,3]
                D_npp[:,5] = nppFAOWh[t-261,:] - npp_0[:,5]
                D_npp[:,6] = nppFAORi[t-261,:] - npp_0[:,6]
                D_npp[:,7] = nppFAOMa[t-261,:] - npp_0[:,7]
                D_npp[:,8] = nppFAOSo[t-261,:] - npp_0[:,8]
            elif t> Ys-1700:
                D_TY = D_lst- D_lst_t[Ys-1700,:]  #the difference between temperature at t and at 2017
                D_CO2_Y = D_CO2_t[Ys-1700] # CO2 at the year 2017
                D_lyp_Y = D_lyp_t[Ys-1700,:] # precipitation at the year 2017
                nppOtY = nppFAOOt[Ys-1961,:]
                nppWhY = nppFAOWh[Ys-1961,:]
                nppRiY = nppFAORi[Ys-1961,:]
                nppMaY = nppFAOMa[Ys-1961,:]
                nppSoY = nppFAOSo[Ys-1961,:]
                
                FP = np.exp(gamma_nppP*D_lyp[:,np.newaxis])/np.exp(gamma_nppP*D_lyp_Y[:,np.newaxis])
                
                for i in range(10):
                    D_npp[i,3] = max(-1*npp_0[i,3],(nppOtY[i]* f_nppcrYStart(FC,FCY,FTReg[:,np.newaxis],FP)[i,3]-npp_0[i,3]))
                    D_npp[i,5] = max(-1*npp_0[i,5],(nppWhY[i]* f_nppcrYStart(FC,FCY,FTReg[:,np.newaxis],FP)[i,5]-npp_0[i,5]))
                    D_npp[i,6] = max(-1*npp_0[i,6],(nppRiY[i]* f_nppcrYStart(FC,FCY,FTReg[:,np.newaxis],FP)[i,6]-npp_0[i,6])) # cropland-cereals npp
                    D_npp[i,7] = max(-1*npp_0[i,7],(nppMaY[i]* f_nppcrYStart(FC_Ma,FCY,FTReg_Ma[:,np.newaxis],FP)[i,7]-npp_0[i,7])) # cropland-cereals npp
                    D_npp[i,8] = max(-1*npp_0[i,8],(nppSoY[i]* f_nppcrYStart(FC,FCY,FTReg[:,np.newaxis],FP)[i,8]-npp_0[i,8])) # cropland-cereals npp

            '''calculate production in model  '''
            NPP_crop = (D_npp[:,3]+npp_0[:,3])*(D_AREA[:,3]+AREA_0[:,3])
            
            Prodwh_cou = Yieldwh_cou*Areawh_cou
            Prodri_cou = Yieldri_cou*Areari_cou
            Prodma_cou = Yieldma_cou*Areama_cou
            Prodso_cou = Yieldso_cou*Areaso_cou
            Prodot_cou = Yieldot_cou*Areaot_cou
            
            ## Total yield
            if t> Ys-1701:
                YieldTot = np.sum(Prodwh_cou+Prodri_cou+Prodma_cou+Prodso_cou)/np.sum(Areawh_cou+Areari_cou+Areama_cou+Areaso_cou)
            
            ##Calculate production per capita per day at country level, note that population is constant at 2020***
            ProdDC_cou = np.zeros([167])
            if t >329:
                ProdDC_cou = (Prodwh_cou+Prodri_cou+Prodma_cou+Prodso_cou)/Pop[0]/365
            
            ## Calculate energy per capita per day
            CalCon = np.array([3392*0.8*0.78,3882*0.89,3623*0.3*0.79]) 
            ResRat = np.zeros([1,167])
            DevCod = np.array([line for line in csv.reader(open('data/'+Filein+'/CodeDevCou.csv','r'))],dtype=dty) #2020-2500
            X1=0.44; X2 = 0.56
            for i in range(167):
                if int(DevCod[0,i]) == 1: # 1- developed countries
                    ResRat[0,i] = X1
                elif int(DevCod[0,i]) == 2: # 1- developing countries
                    ResRat[0,i] = X2  
            AniCal = np.array([line for line in csv.reader(open('data/'+Filein+'/AnimalCal.csv','r'))],dtype=dty).T  # kcal/capita/day
            for n in range(167):
                if AniCal[0,n]<250:
                    AniCal[0,n] = 250
                if AniCal[0,n]>1000:
                    AniCal[0,n] = 1000
            
            SumAniCal = 1000*365*AniCal*Pop[0,:] #kcal
                    
            if t >319:
                ProdGlobalCap = np.sum(Prodwh_cou+Prodri_cou+Prodma_cou+Prodso_cou)/np.sum(Pop[0])/365
                EnerDCGlo = 10**9*np.sum(Prodwh_cou*ResRat*CalCon[0]+Prodri_cou*ResRat*CalCon[1]+Prodma_cou*ResRat*CalCon[2]+(10**-12)*SumAniCal)/np.sum(Pop[0])/365
            
            ## Calculate developing countries data  
                ProdDCTotal = Prodwh_cou+Prodri_cou+Prodma_cou+Prodso_cou
                ProdDC = np.array([0],dtype=dty)
                PopDC = np.array([0],dtype=dty)
                for c in range(167):
                    if CodeDC[0,c] == 2: # 2- not developed country
                        ProdDC += ProdDCTotal[c]
                        PopDC += Pop[0,c]
                ProdDCGloCap = ProdDC/PopDC/365
                
#            #Add BECCS
            BECCSall = np.zeros([1],dtype=dty)								
            ''' BECCS, choose to open or close''' 											
            if t > int(Ybeccs)-1700 - 1:
                ###  consider only the first part multiply water content
                BECCSall =np.sum(0.8125*Prodwh_cou[:,np.newaxis] + 0.7188*Prodri_cou[:,np.newaxis] + 0.7659*Prodma_cou[:,np.newaxis] + 1.2523*Prodso_cou[:,np.newaxis] + Fbe*Prodot_cou[:,np.newaxis])

            D_efire = igni_0 * ((1+D_k_igni)*(cveg_0 + D_cveg) - cveg_0)
            D_fmort = mu_0 * D_cveg
            D_rh1 = rho1_0 * ((1+D_k_rho)*(csoil1_0 + D_csoil1) - csoil1_0)
            D_fmet = k_met * D_rh1
            D_rh2 = rho2_0 * ((1+D_k_rho)*(csoil2_0 + D_csoil2) - csoil2_0)
            D_ewet = ewet_0 * np.nan_to_num(np.sum(p_wet * D_csoil1,1) / np.sum(p_wet * csoil1_0,1))
            
            bLSNK = D_rh1 + D_rh2 + D_efire - D_npp
            bLSNK[:,3] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.34447*D_npp)[:,3]
            bLSNK[:,5] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.34701208*D_npp)[:,5]
            bLSNK[:,6] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.35408804*D_npp)[:,6]
            bLSNK[:,7] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.370221514*D_npp)[:,7]
            bLSNK[:,8] = (D_rh1 + D_rh2 + D_efire - D_npp + 0.306569343*D_npp)[:,8]
            
            LSNK = np.sum((D_rh1 + D_rh2 + D_efire - D_npp)*(AREA_0 + D_AREA)) 
            
            # in terms of biomes                                                                                                                                                
            D_EWET = ewet_0*D_AWET + D_ewet*AWET_0 + D_ewet*D_AWET 
            # stocks
            D_cveg += (p**-1) * (D_npp - D_fmort - D_efire)
            D_csoil1 += (p**-1) * (D_fmort - D_fmet - D_rh1)
            D_csoil2 += (p**-1) * (D_fmet - D_rh2)
            
            Diff = abs(D_rh1) + abs(D_rh2) - abs(D_npp)
            
            # 3. LAND-USE
            #------------
            
            # initialization
            # land-use change
            for b1 in range(nb_biome):
                for b2 in range(nb_biome):
                    CVEG_luc[:,b1,b2,t] +=  (p**-1) * -(cveg_0+D_cveg)[:,b2] * LUC[t,:,b1,b2] 
                    CSOIL1_luc[:,b1,b2,t] += (p**-1) * ((csoil1_0+D_csoil1)[:,b1] - (csoil1_0+D_csoil1)[:,b2]) * LUC[t,:,b1,b2]
                    CSOIL2_luc[:,b1,b2,t] += (p**-1) * ((csoil2_0+D_csoil2)[:,b1] - (csoil2_0+D_csoil2)[:,b2]) * LUC[t,:,b1,b2]
                    CSOIL1_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (p_AGB[:,b1]*p_HWP0[:,b1]+(1-p_AGB[:,b1])) * LUC[t,:,b1,b2]
                    CHWP1_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * p_AGB[:,b1]*p_HWP1[:,b1] * LUC[t,:,b1,b2]
                    CHWP2_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * p_AGB[:,b1]*p_HWP2[:,b1] * LUC[t,:,b1,b2]
                    CHWP3_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * p_AGB[:,b1]*p_HWP3[:,b1] * LUC[t,:,b1,b2]
            # harvest
            for b in range(nb_biome):
                CVEG_luc[:,b,b,t] += (p**-1) * -HARV[t,:,b]
                CSOIL1_luc[:,b,b,t] += (p**-1) * p_HWP0[:,b] * HARV[t,:,b]
                CHWP1_luc[:,b,b,t] += (p**-1) * p_HWP1[:,b] * HARV[t,:,b]
                CHWP2_luc[:,b,b,t] += (p**-1) * p_HWP2[:,b] * HARV[t,:,b]
                CHWP3_luc[:,b,b,t] += (p**-1) * p_HWP3[:,b] * HARV[t,:,b]
            # shifting cultivation
            for b1 in range(nb_biome):
                for b2 in range(b1,nb_biome):
                    CVEG_luc[:,b1,b2,t] += (p**-1) * -(cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * SHIFT[t,:,b1,b2]
                    CSOIL1_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * (p_AGB[:,b1]*p_HWP0[:,b1]+(1-p_AGB[:,b1])) * SHIFT[t,:,b1,b2]
                    CHWP1_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * p_AGB[:,b1]*p_HWP1[:,b1] * SHIFT[t,:,b1,b2]
                    CHWP2_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * p_AGB[:,b1]*p_HWP2[:,b1] * SHIFT[t,:,b1,b2]
                    CHWP3_luc[:,b1,b2,t] += (p**-1) * (cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * p_AGB[:,b1]*p_HWP3[:,b1] * SHIFT[t,:,b1,b2]
                    CVEG_luc[:,b2,b1,t] += (p**-1) * -(cveg_0+D_cveg)[:,b1] * (1-np.exp(-mu_0[:,b1]*tau_shift)) * SHIFT[t,:,b1,b2]
                    CSOIL1_luc[:,b2,b1,t] += (p**-1) * (cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * (p_AGB[:,b2]*p_HWP0[:,b2]+(1-p_AGB[:,b2])) * SHIFT[t,:,b1,b2]
                    CHWP1_luc[:,b2,b1,t] += (p**-1) * (cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * p_AGB[:,b2]*p_HWP1[:,b2] * SHIFT[t,:,b1,b2]
                    CHWP2_luc[:,b2,b1,t] += (p**-1) * (cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * p_AGB[:,b2]*p_HWP2[:,b2] * SHIFT[t,:,b1,b2]
                    CHWP3_luc[:,b2,b1,t] += (p**-1) * (cveg_0+D_cveg)[:,b2] * (1-np.exp(-mu_0[:,b2]*tau_shift)) * p_AGB[:,b2]*p_HWP3[:,b2] * SHIFT[t,:,b1,b2]

            # fluxes
            # book-keeping model
            NPP_luc = 0*CVEG_luc
            EFIRE_luc = (igni_0*(1+D_k_igni))[:,np.newaxis,:,np.newaxis] * CVEG_luc
            FMORT_luc = mu_0[:,np.newaxis,:,np.newaxis] * CVEG_luc
            RH1_luc = (rho1_0*(1+D_k_rho))[:,np.newaxis,:,np.newaxis] * CSOIL1_luc
            FMET_luc = k_met * RH1_luc
            RH2_luc = (rho2_0*(1+D_k_rho))[:,np.newaxis,:,np.newaxis] * CSOIL2_luc
            EHWP1_luc = np.zeros([nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty)
            EHWP1_luc[:,:,:,:t+1] = (r_HWP1*(1-r_HWP1**tt))[np.newaxis,np.newaxis,np.newaxis,t::-1] * CHWP1_luc[:,:,:,:t+1]
            EHWP2_luc = np.zeros([nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty)
            EHWP2_luc[:,:,:,:t+1] = (r_HWP2*(1-r_HWP2**tt))[np.newaxis,np.newaxis,np.newaxis,t::-1] * CHWP2_luc[:,:,:,:t+1]
            EHWP3_luc = np.zeros([nb_regionI,nb_biome,nb_biome,ind_final+1],dtype=dty) 
            EHWP3_luc[:,:,:,:t+1] = (r_HWP3*(1-r_HWP3**tt))[np.newaxis,np.newaxis,np.newaxis,t::-1] * CHWP3_luc[:,:,:,:t+1]
            ELUC = np.sum(np.sum(np.sum( RH1_luc + RH2_luc + EFIRE_luc + EHWP1_luc + EHWP2_luc + EHWP3_luc ,3),2),1)
            b1ELUC = np.sum(np.sum( RH1_luc + RH2_luc + EFIRE_luc + EHWP1_luc + EHWP2_luc + EHWP3_luc ,3),2)
            b2ELUC = np.sum(np.sum( RH1_luc + RH2_luc + EFIRE_luc + EHWP1_luc + EHWP2_luc + EHWP3_luc ,3),1)
            b1RH1_luc =  np.sum(np.sum( RH1_luc,3),2)
            b2RH1_luc =  np.sum(np.sum( RH1_luc,3),1)
            b1RH2_luc =  np.sum(np.sum( RH2_luc,3),2)
            b2RH2_luc =  np.sum(np.sum( RH2_luc,3),1)
            b1EFIRE_luc = np.sum(np.sum( EFIRE_luc,3),2)
            b2EFIRE_luc = np.sum(np.sum( EFIRE_luc,3),1)
            b1EHWP1_luc = np.sum(np.sum( EHWP1_luc,3),2)
            b2EHWP1_luc = np.sum(np.sum( EHWP1_luc,3),1)
            b1EHWP2_luc = np.sum(np.sum( EHWP2_luc,3),2)
            b2EHWP2_luc = np.sum(np.sum( EHWP2_luc,3),1)
            b1EHWP3_luc = np.sum(np.sum( EHWP3_luc,3),2)
            b2EHWP3_luc = np.sum(np.sum( EHWP3_luc,3),1)

            #biomass burning
            for VAR in ['CO2','CH4','N2O','NOX','CO','VOC','SO2','NH3','OC','BC']:
                exec('D_EBB_'+VAR+' = np.sum( alpha_BB_'+VAR+'*(igni_0*cveg_0*D_AREA + D_efire*AREA_0 + D_efire*D_AREA) ,1)')
                exec('D_EBB_'+VAR+' += np.sum(np.sum(np.sum( alpha_BB_'+VAR+'[:,:,np.newaxis,np.newaxis]*EHWP1_luc ,3),2),1)')
                exec('D_EBB_'+VAR+' += np.sum(np.sum(np.sum( alpha_BB_'+VAR+'[:,np.newaxis,:,np.newaxis]*EFIRE_luc ,3),2),1)')
                           
            # stocks
            CVEG_luc += (p**-1) * (NPP_luc - FMORT_luc - EFIRE_luc)
            CSOIL1_luc += (p**-1) * (FMORT_luc - FMET_luc - RH1_luc)
            CSOIL2_luc += (p**-1) * (FMET_luc - RH2_luc)
            CHWP1_luc += (p**-1) * -EHWP1_luc
            CHWP2_luc += (p**-1) * -EHWP2_luc
            CHWP3_luc += (p**-1) * -EHWP3_luc
            #-------------
            # 4. CHEMISTRY
            #-------------

            # factors
            D_kOH = f_kOH(D_CH4,D_O3s,fT*D_gst,np.sum(ENOX[t]+D_EBB_NOX),np.sum(ECO[t]+D_EBB_CO),np.sum(EVOC[t]+D_EBB_VOC))
            D_hv = f_hv(D_N2O_lag,D_EESC,fT*D_gst)
            # fluxes
            D_OHSNK_CH4 = -alpha_CH4/tau_CH4_OH * (CH4_0*D_kOH + D_CH4 + D_kOH*D_CH4)
            D_HVSNK_CH4 = -alpha_CH4/tau_CH4_hv * (CH4_0*D_hv + D_CH4_lag + D_hv*D_CH4_lag)
            D_XSNK_CH4 = -alpha_CH4*(1/tau_CH4_soil + 1/tau_CH4_ocean) * D_CH4
            D_HVSNK_N2O = -alpha_N2O/tau_N2O_hv * (N2O_0*D_hv + D_N2O_lag + D_hv*D_N2O_lag)
            for VAR in ['HFC','PFC','ODS']:
                exec('D_OHSNK_'+VAR+' = -alpha_'+VAR+'/tau_'+VAR+'_OH * ('+VAR+'_0*D_kOH + D_'+VAR+' + D_kOH*D_'+VAR+')')
                exec('D_HVSNK_'+VAR+' = -alpha_'+VAR+'/tau_'+VAR+'_hv * ('+VAR+'_0*D_hv + D_'+VAR+'_lag + D_hv*D_'+VAR+'_lag)')
                exec('D_XSNK_'+VAR+' = -alpha_'+VAR+'/tau_'+VAR+'_othr * D_'+VAR)
            # stocks
            D_O3t = chi_O3t_CH4*np.log(1+D_CH4/CH4_0) + Gamma_O3t*fT*D_gst
            D_O3t += chi_O3t_NOX*np.sum(w_reg_NOX*np.sum(p_reg4*(ENOX[t]+D_EBB_NOX)[:,np.newaxis],0))
            D_O3t += chi_O3t_CO*np.sum(w_reg_CO*np.sum(p_reg4*(ECO[t]+D_EBB_CO)[:,np.newaxis],0))
            D_O3t += chi_O3t_VOC*np.sum(w_reg_VOC*np.sum(p_reg4*(EVOC[t]+D_EBB_VOC)[:,np.newaxis],0))
            D_EESC = np.sum(f_fracrel(tau_lag) * (n_Cl+alpha_Br*n_Br) * D_ODS_lag)
            D_O3s = chi_O3s_EESC*D_EESC + chi_O3s_N2O*D_N2O_lag * (1-D_EESC/EESC_x) + Gamma_O3s*fT*D_gst
            D_SO4 = alpha_SO4*tau_SO2*np.sum(w_reg_SO2*np.sum(p_reg4*(ESO2[t]+D_EBB_SO2)[:,np.newaxis],0)) + alpha_SO4*tau_DMS*0 + Gamma_SO4*fT*D_gst
            D_POA = tau_OMff*alpha_POM*np.sum(w_reg_OC*np.sum(p_reg4*(EOC[t])[:,np.newaxis],0)) + tau_OMbb*alpha_POM*np.sum(D_EBB_OC) + Gamma_POA*fT*D_gst
            D_BC = tau_BCff*np.sum(w_reg_BC*np.sum(p_reg4*(EBC[t])[:,np.newaxis],0)) + tau_BCbb*np.sum(D_EBB_BC) + Gamma_BC*fT*D_gst
            D_NO3 = alpha_NO3*tau_NOX*np.sum(ENOX[t]+D_EBB_NOX) + alpha_NO3*tau_NH3*np.sum(ENH3[t]+D_EBB_NH3) + Gamma_NO3*fT*D_gst
            D_SOA = tau_VOC*np.sum(EVOC[t]+D_EBB_VOC) + tau_BVOC*0 + Gamma_SOA*fT*D_gst
            D_DUST = 0*( tau_DUST*0 + Gamma_DUST*fT*D_gst )
            D_SALT = 0*( tau_SALT*0 + Gamma_SALT*fT*D_gst )
            D_AERh = solub_SO4*D_SO4 + solub_POA*D_POA + solub_BC*D_BC + solub_NO3*D_NO3 + solub_SOA*D_SOA + solub_DUST*D_DUST + solub_SALT*D_SALT
            
            #--------------
            # 5. ATMOSPHERE
            #--------------

            # stocks
            # Make EFF = 0 after 2020
            
            if t < Ybeccs-1700:
                D_CO2 += (p**-1) * (1/alpha_CO2) * (np.sum(EFF[t]) + np.sum(ELUC) + LSNK + OSNK - BECCSall + Emi_exp[t])
            elif t > Ybeccs-1700 :
                if D_CO2 < Cm:
                    D_CO2 = (Cm-Cn)*D_CO2**2/(Cm**2) + Cn
                else:
                    D_CO2 += (p**- 1) * (1/alpha_CO2) * (np.sum(EFF[t]) + np.sum(ELUC) + LSNK + OSNK - BECCSall + Emi_exp[t])
                    
            if t <321:
                D_N2O += (p**-1) * (1/alpha_N2O) * (np.sum(EN2O[t]) + np.sum(D_EBB_N2O) + D_HVSNK_N2O)
            elif t < Tf+1:
                D_N2O = Pf + (D_N2O_t[320]-Pf)*((t-Tf)**2)/((320-Tf)**2)
            else:
                D_N2O = Pf + (D_N2O_t[Tf]-Pf)*((t-Tf)**2)/((320-Tf)**2)
            
            if t<321:
                D_CH4 += (p**-1) * (1/alpha_CH4) * (np.sum(ECH4[t]) + np.sum(D_EBB_CH4) + np.sum(D_EWET) + D_OHSNK_CH4 + D_HVSNK_CH4 + D_XSNK_CH4)
            else:
                D_CH4 = 1500 + (D_CH4_t[320]-1500)*((t-800)**2)/(480**2)
            
            D_HFC += (p**-1) * (1/alpha_HFC) * (np.sum(EHFC[t],0) + D_OHSNK_HFC + D_HVSNK_HFC + D_XSNK_HFC)
            D_PFC += (p**-1) * (1/alpha_PFC) * (np.sum(EPFC[t],0) + D_OHSNK_PFC + D_HVSNK_PFC + D_XSNK_PFC)
            D_ODS += (p**-1) * (1/alpha_ODS) * (np.sum(EODS[t],0) + D_OHSNK_ODS + D_HVSNK_ODS + D_XSNK_ODS)
            for VAR in ['CH4','N2O','HFC','PFC','ODS']:
                exec('D_'+VAR+'_lag += (p**-1) * ((1/tau_lag)*D_'+VAR+' - (1/tau_lag)*D_'+VAR+'_lag)')

            # FORCE
            if force_CO2:
                D_CO2 = D_CO2_force[t]
            
            if force_GHG:
                D_CO2 = D_CO2_force[t]
                D_CH4 = D_CH4_force[t]
                D_N2O = D_N2O_force[t]
            
            if force_halo:
                D_HFC[:] = D_HFC_force[t]
                D_PFC[:] = D_PFC_force[t]
                D_ODS[:] = D_ODS_force[t]
            
            #-----------
            # 6. CLIMATE
            #-----------

            # fluxes
            # per component
            RF_CO2 = f_RF_CO2(D_CO2)
            RF_CH4 = f_RF_CH4(D_CH4)-(f_RF_overlap(D_CH4,D_N2O)-f_RF_overlap(0,D_N2O))
            RF_H2Os = f_RF_H2Os(D_CH4_lag)
            RF_N2O = f_RF_N2O(D_N2O)-(f_RF_overlap(D_CH4,D_N2O)-f_RF_overlap(D_CH4,0))
            RF_halo = np.sum(radeff_HFC*D_HFC) + np.sum(radeff_PFC*D_PFC) + np.sum(radeff_ODS*D_ODS)
            for VAR in ['O3t','O3s','SO4','POA','BC','NO3','SOA','DUST','SALT']:
                exec('RF_'+VAR+' = radeff_'+VAR+'*D_'+VAR)
            RF_cloud = k_BC_adjust*RF_BC + Phi_0*np.log(1+max(-0.9,D_AERh/AERh_0))
            RF_BCsnow = radeff_BCsnow*np.sum(w_reg_BCsnow*np.sum(p_reg9*(EBC[t]+D_EBB_BC)[:,np.newaxis],0))
            RF_LCC = np.sum(alpha_LCC*D_AREA)
            if t > 317:
                RF_LCC = np.sum(alpha_LCC*D_AREA_t[317])
                
            # FORCE
            if force_RFs:
                for VAR in ['CO2','CH4','H2Os','N2O','halo']+['O3t','O3s','SO4','POA','BC','NO3','SOA','DUST','SALT']+['cloud','BCsnow','LCC']:
                    exec('RF_'+VAR+' = RF_'+VAR+'_force[t]')

            # totals
            RF = RF_CO2 + RF_CH4 + RF_H2Os + RF_N2O + RF_halo + RF_O3t + RF_O3s + RF_SO4 + RF_POA + RF_BC + RF_NO3 + RF_SOA + RF_DUST + RF_SALT + RF_cloud + RF_BCsnow + RF_LCC + RFcon[t] + RFvolc[t] + RFsolar[t]
            RF_warm = RF_CO2 + RF_CH4 + RF_H2Os + RF_N2O + RF_halo + RF_O3t + RF_O3s + RF_SO4 + RF_POA + RF_BC + RF_NO3 + RF_SOA + RF_DUST + RF_SALT + RF_cloud + warmeff_BCsnow*RF_BCsnow + warmeff_LCC*RF_LCC + RFcon[t] + warmeff_volc*RFvolc[t] + RFsolar[t]
            RF_atm = p_atm_CO2*RF_CO2 + p_atm_noCO2*(RF_CH4+RF_N2O+RF_halo) + p_atm_O3t*RF_O3t + p_atm_strat*(RF_O3s+RF_H2Os) + p_atm_scatter*(RF_SO4+RF_POA+RF_NO3+RF_SOA+RF_DUST+RF_SALT+RFvolc[t]) + p_atm_absorb*RF_BC + p_atm_cloud*(RF_cloud+RFcon[t]) + p_atm_alb*(RF_BCsnow+RF_LCC) + p_atm_solar*RFsolar[t]

            # FORCE
            if force_RF:
                RF_warm = RF_force[t] * (RF_warm/RF)
                RF_atm = RF_force[t] * (RF_atm/RF)
                RF = RF_force[t]

            # stocks
            # temperatures
            D_gst += (p**-1) * (1/tau_gst) * (lambda_0*RF_warm - D_gst - theta_0*(D_gst-D_gst0))
            D_gst0 += (p**-1) * (1/tau_gst0) * theta_0*(D_gst-D_gst0)
            D_sst = w_reg_sst*D_gst
            D_lst = w_reg_lst*D_gst
            # precipitations
            D_gyp = alpha_gyp*D_gst + beta_gyp*RF_atm
            D_lyp = w_reg_lyp*D_gyp
            # ocean
            D_OHC += (p**-1) * p_OHC * alpha_OHC * (RF - D_gst/lambda_0)
            D_pH = f_pH(D_CO2)

            # FORCE
            if force_clim:
                D_gst = D_gst_force[t]
                D_sst = D_sst_force[t]
                D_lst = D_lst_force[t]
                D_lyp = D_lyp_force[t]

            #-----------
            # Y. SAVE
            #-----------
            for var in var_timeseries:
                exec(var+'_t[t] += (p**-1) * '+var)

            #---------
            # Z. TESTS
            #---------
        
            if np.isnan(np.sum(D_CO2)):
                print 'D_CO2 = NaN at t = '+str(t)+' and tt = '+str(tt)
                print 'OSNK = '+str(np.sum(OSNK))
                print 'LSNK = '+str(np.sum(LSNK))
                print 'ELUC = '+str(np.sum(ELUC))
                break
            if np.isnan(np.sum(D_CH4)):
                print 'D_CH4 = NaN at t = '+str(t)+' and tt = '+str(tt)              
                print 'D_EWET = '+str(np.sum(D_EWET))
                print 'D_OHSNK = '+str(np.sum(D_OHSNK_CH4))
                print 'D_HVSNK = '+str(np.sum(D_HVSNK_CH4))
                break
            if np.isnan(np.sum(D_gst)):
                print 'D_gst = NaN at t = '+str(t)+' and tt = '+str(tt)
                print 'RF_CO2 = '+str(np.sum(RF_CO2))
                print 'RF_CH4 = '+str(np.sum(RF_CH4))
                print 'RF_H2Os = '+str(np.sum(RF_H2Os))
                print 'RF_N2O = '+str(np.sum(RF_N2O))
                print 'RF_halo = '+str(np.sum(RF_halo))
                print 'RF_O3t = '+str(np.sum(RF_O3t))
                print 'RF_O3s = '+str(np.sum(RF_O3s))
                print 'RF_SO4 = '+str(np.sum(RF_SO4))
                print 'RF_POA = '+str(np.sum(RF_POA))
                print 'RF_BC = '+str(np.sum(RF_BC))
                print 'RF_NO3 = '+str(np.sum(RF_NO3))
                print 'RF_SOA = '+str(np.sum(RF_SOA))
                print 'RF_DUST = '+str(np.sum(RF_DUST))
                print 'RF_SALT = '+str(np.sum(RF_SALT))
                print 'RF_cloud = '+str(np.sum(RF_cloud))
                print 'RF_BCsnow = '+str(np.sum(RF_BCsnow))
                print 'RF_LCC = '+str(np.sum(RF_LCC))
                break

        if np.isnan(np.sum(D_CO2))|np.isnan(np.sum(D_CH4))|np.isnan(np.sum(D_gst)):
            for var in var_timeseries:
                if (t < ind_final):
                    exec(var+'_t[t+1:] = np.nan')
            break

    #===========
    # C. FIGURES
    #===========

    if plot is 'all' or plot is 'CO2' or 'CO2' in plot:
        plot_CO2(D_CO2_t,OSNK_t,LSNK_t,ELUC_t,EFF,D_AREA_t,D_npp_t,D_efire_t,D_fmort_t,D_rh1_t,D_fmet_t,D_rh2_t,D_FIN_t,D_FOUT_t,D_FCIRC_t,EFIRE_luc_t,FMORT_luc_t,RH1_luc_t,RH2_luc_t,EHWP1_luc_t,EHWP2_luc_t,EHWP3_luc_t)
    if plot is 'all' or plot is 'CH4' or 'CH4' in plot:
        plot_CH4(D_CH4_t,D_OHSNK_CH4_t,D_HVSNK_CH4_t,D_XSNK_CH4_t,D_EWET_t,D_EBB_CH4_t,ECH4)
    if plot is 'all' or plot is 'N2O' or 'N2O' in plot:
        plot_N2O(D_N2O_t,D_HVSNK_N2O_t,D_EBB_N2O_t,EN2O)
    if plot is 'all' or plot is 'O3' or 'O3' in plot:
        plot_O3(D_O3t_t,D_O3s_t,D_EESC_t,D_N2O_lag_t,D_gst_t)
    if plot is 'all' or plot is 'AER' or 'AER' in plot:
        plot_AER(D_SO4_t,D_POA_t,D_BC_t,D_NO3_t,D_SOA_t,D_AERh_t,RF_SO4_t,RF_POA_t,RF_BC_t,RF_NO3_t,RF_SOA_t,RF_cloud_t)
    if plot is 'all' or plot is 'clim' or 'clim' in plot:
        plot_clim(RF_t,D_gst_t,D_gyp_t,RF_CO2_t,RF_CH4_t,RF_H2Os_t,RF_N2O_t,RF_halo_t,RF_O3t_t,RF_O3s_t,RF_SO4_t,RF_POA_t,RF_BC_t,RF_NO3_t,RF_SOA_t,RF_cloud_t,RF_BCsnow_t,RF_LCC_t,RFcon,RFvolc,RFsolar)

    #===========
    # D. OUTPUTS
    #===========

    output = []
    for var in var_output:
        exec('output.append('+var+'_t)')      

    return output

   
##################################################
#   2. CONTROL PLOTS
##################################################

#=========
# 2.1. CO2
#=========

def plot_CO2(D_CO2,OSNK,LSNK,ELUC,EFF,D_AREA,D_npp,D_efire,D_fmort,D_rh1,D_fmet,D_rh2,D_FIN,D_FOUT,D_FCIRC,D_MORT_luc,D_EFIRE_luc,D_RH1_luc,D_RH2_luc,EHWP1_luc,EHWP2_luc,EHWP3_luc):
    plt.figure()

    # atmospheric CO2
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_CO2,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(CO2_ipcc)),CO2_ipcc-CO2_0,color='r',lw=2,ls='--',label='IPCC')    
    if (ind_final > ind_cdiac):
        plt.plot(1700+np.arange(min(len(CO2_rcp),ind_final+1)),CO2_rcp[:min(len(CO2_rcp),ind_final+1),0]-CO2_0,color='0.8',lw=2,ls=':',label='RCP2.6')
        plt.plot(1700+np.arange(min(len(CO2_rcp),ind_final+1)),CO2_rcp[:min(len(CO2_rcp),ind_final+1),1]-CO2_0,color='0.6',lw=2,ls=':',label='RCP4.5')
        plt.plot(1700+np.arange(min(len(CO2_rcp),ind_final+1)),CO2_rcp[:min(len(CO2_rcp),ind_final+1),2]-CO2_0,color='0.4',lw=2,ls=':',label='RCP6.0')
        plt.plot(1700+np.arange(min(len(CO2_rcp),ind_final+1)),CO2_rcp[:min(len(CO2_rcp),ind_final+1),3]-CO2_0,color='0.2',lw=2,ls=':',label='RCP8.5')
    plt.title('$\Delta$CO2 (ppm)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # budget fluxes
    ax = plt.subplot(2,3,2)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),np.sum(EFF,1),color='#666666',lw=2,label='EFF')
    plt.plot(1700+np.arange(ind_final+1),np.sum(ELUC,1),color='#993300',lw=2,label='ELUC')
    plt.plot(1700+np.arange(ind_final+1),OSNK,color='#000099',lw=2,label='OSNK')
    plt.plot(1700+np.arange(ind_final+1),LSNK,color='#009900',lw=2,label='LSNK')
    plt.plot(1700+np.arange(ind_final)+1,alpha_CO2*(D_CO2[1:]-D_CO2[:-1]),color='#FFCC00',lw=2,label='d_CO2')
    plt.plot(1700+np.arange(len(EFF_gcp)),EFF_gcp,color='#666666',ls='--')
    plt.plot(1700+np.arange(len(ELUC_gcp)),ELUC_gcp,color='#CC3300',ls='--')
    plt.plot(1700+np.arange(len(OSNK_gcp)),OSNK_gcp,color='#000099',ls='--')
    plt.plot(1700+np.arange(len(LSNK_gcp)),LSNK_gcp,color='#009900',ls='--')
    plt.plot(1700+np.arange(len(d_CO2_gcp)),d_CO2_gcp,color='#FFCC00',ls='--')
    plt.plot([1700,1700],[0,0],'k--',label='GCP')
    plt.title('CO2 fluxes (GtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # airborne fraction
    ax = plt.subplot(2,3,3)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final)+1,alpha_CO2*(D_CO2[1:]-D_CO2[:-1])/np.sum(EFF+ELUC,1)[1:],color='#FFCC00',lw=1,label='AF')
    plt.plot(1700+np.arange(ind_final+1),-OSNK/np.sum(EFF+ELUC,1),color='#000099',lw=1,label='OF')
    plt.plot(1700+np.arange(ind_final+1),-LSNK/np.sum(EFF+ELUC,1),color='#009900',lw=1,label='LF')    
    plt.plot(np.arange(1959,1700+ind_cdiac+1),np.ones([ind_cdiac-259+1])*np.mean((alpha_CO2*(D_CO2[1:]-D_CO2[:-1])/np.sum(EFF+ELUC,1)[1:])[259-1:ind_cdiac]),color='k',lw=2,label='OSCAR')
    plt.plot(np.arange(1959,1700+ind_cdiac+1),np.ones([ind_cdiac-259+1])*np.mean((d_CO2_gcp/(EFF_gcp+ELUC_gcp))[259:ind_cdiac+1]),color='r',lw=2,ls='--',label='GCP')
    plt.title('airborne fraction',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))             
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])
    ax.set_ylim([-0.2,1.2])

    # ELUC details
    ax = plt.subplot(2,3,4)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),np.sum(ELUC,1),color='k',ls='-.',lw=2,label='ELUC')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(D_EFIRE_luc+D_RH1_luc+D_RH2_luc,4),3),2),1),color='#009900',lw=2,label='ELUC_bio')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(EHWP1_luc+EHWP2_luc+EHWP3_luc,4),3),2),1),color='#993300',lw=2,label='ELUC_hwp')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(EHWP1_luc,4),3),2),1),color='#FF3300',lw=1,label='EHWP1')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(EHWP2_luc,4),3),2),1),color='#CC9900',lw=1,label='EHWP2')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(np.sum(np.sum(EHWP3_luc,4),3),2),1),color='#663300',lw=1,label='EHWP3')    
    plt.title('ELUC fluxes (GtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))

    # LSNK details
    ax = plt.subplot(2,3,5)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),-LSNK,color='k',lw=2,ls='-.',label='$-$LSNK')    
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(D_npp*(AREA_0+D_AREA),2),1),color='#009900',lw=2,label='D_NPP')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(D_efire*(AREA_0+D_AREA),2),1),color='#FF3300',lw=2,label='D_EFIRE')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum(D_fmort*(AREA_0+D_AREA),2),1),color='#336633',lw=2,label='D_FMORT')
    plt.plot(1700+np.arange(ind_final+1),np.sum(np.sum((D_rh1+D_rh2)*(AREA_0+D_AREA),2),1),color='#663300',lw=2,label='D_RH')
    plt.title('LSNK fluxes (GtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))

    # OSNK details
    ax = plt.subplot(2,3,6)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),-OSNK,color='k',lw=2,ls='-.',label='$-$OSNK')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_FIN,1),color='#000099',lw=2,label='D_FIN')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_FOUT,1),color='#0099FF',lw=2,label='D_FOUT')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_FCIRC,1),color='#663399',lw=2,label='D_FCIRC')             
    plt.title('OSNK fluxes (GtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))

#=========
# 2.2. CH4
#=========

def plot_CH4(D_CH4,D_OHSNK_CH4,D_HVSNK_CH4,D_XSNK_CH4,D_EWET,D_EBB_CH4,ECH4):
    plt.figure()

    # atmospheric CH4
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_CH4,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(CH4_ipcc)),CH4_ipcc-CH4_0,color='r',lw=2,ls='--',label='IPCC')    
    if (ind_final > ind_cdiac):
        plt.plot(1700+np.arange(min(len(CH4_rcp),ind_final+1)),CH4_rcp[:min(len(CH4_rcp),ind_final+1),0]-CH4_0,color='0.8',lw=2,ls=':',label='RCP2.6')
        plt.plot(1700+np.arange(min(len(CH4_rcp),ind_final+1)),CH4_rcp[:min(len(CH4_rcp),ind_final+1),1]-CH4_0,color='0.6',lw=2,ls=':',label='RCP4.5')
        plt.plot(1700+np.arange(min(len(CH4_rcp),ind_final+1)),CH4_rcp[:min(len(CH4_rcp),ind_final+1),2]-CH4_0,color='0.4',lw=2,ls=':',label='RCP6.0')
        plt.plot(1700+np.arange(min(len(CH4_rcp),ind_final+1)),CH4_rcp[:min(len(CH4_rcp),ind_final+1),3]-CH4_0,color='0.2',lw=2,ls=':',label='RCP8.5')
    plt.title('$\Delta$CH4 (ppb)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # budget fluxes
    ax = plt.subplot(2,3,2)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),np.sum(ECH4,1),color='#666666',lw=2,label='ECH4')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_EBB_CH4,1),color='#993300',lw=2,label='D_EBB')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_EWET,1),color='#006666',lw=2,label='D_EWET')
    plt.plot(1700+np.arange(ind_final+1),(D_OHSNK_CH4+D_HVSNK_CH4+D_XSNK_CH4),color='#990066',lw=2,label='D_SNK')
    plt.plot(1700+np.arange(ind_final)+1,alpha_CH4*(D_CH4[1:]-D_CH4[:-1]),color='#FFCC00',lw=2,label='d_CH4')
    plt.title('CH4 fluxes (MtC/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # lifetime
    ax = plt.subplot(2,3,3)
    plt.plot(1700+np.arange(ind_final+1),alpha_CH4*(CH4_0+D_CH4)/(alpha_CH4*CH4_0*(1/tau_CH4_OH+1/tau_CH4_hv+1/tau_CH4_soil+1/tau_CH4_ocean)-D_OHSNK_CH4-D_HVSNK_CH4-D_XSNK_CH4),color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(ind_final+1),alpha_CH4*(CH4_0+D_CH4)/(alpha_CH4*CH4_0/tau_CH4_OH-D_OHSNK_CH4),color='k',lw=1,label='OH only')
    plt.title('CH4 lifetime (yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # wetlands
    ax = plt.subplot(2,3,4)
    plt.title('wetlands',fontsize='medium')
 
    # biomass burning
    ax = plt.subplot(2,3,5)
    plt.title('biomass burning',fontsize='medium')


#=========
# 2.3. N2O
#=========

def plot_N2O(D_N2O,D_HVSNK_N2O,D_EBB_N2O,EN2O):
    plt.figure()

    # atmospheric N2O
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_N2O,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(N2O_ipcc)),N2O_ipcc-N2O_0,color='r',lw=2,ls='--',label='IPCC')    
    if (ind_final > ind_cdiac):
        plt.plot(1700+np.arange(min(len(N2O_rcp),ind_final+1)),N2O_rcp[:min(len(N2O_rcp),ind_final+1),0]-N2O_0,color='0.8',lw=2,ls=':',label='RCP2.6')
        plt.plot(1700+np.arange(min(len(N2O_rcp),ind_final+1)),N2O_rcp[:min(len(N2O_rcp),ind_final+1),1]-N2O_0,color='0.6',lw=2,ls=':',label='RCP4.5')
        plt.plot(1700+np.arange(min(len(N2O_rcp),ind_final+1)),N2O_rcp[:min(len(N2O_rcp),ind_final+1),2]-N2O_0,color='0.4',lw=2,ls=':',label='RCP6.0')
        plt.plot(1700+np.arange(min(len(N2O_rcp),ind_final+1)),N2O_rcp[:min(len(N2O_rcp),ind_final+1),3]-N2O_0,color='0.2',lw=2,ls=':',label='RCP8.5')
    plt.title('$\Delta$N2O (ppb)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # budget fluxes
    ax = plt.subplot(2,3,2)
    plt.plot([1700,1700+ind_final+1],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),np.sum(EN2O,1),color='#666666',lw=2,label='EN2O')
    plt.plot(1700+np.arange(ind_final+1),np.sum(D_EBB_N2O,1),color='#993300',lw=2,label='D_EBB')
    plt.plot(1700+np.arange(ind_final+1),D_HVSNK_N2O,color='#990066',lw=2,label='D_SNK')
    plt.plot(1700+np.arange(ind_final)+1,alpha_N2O*(D_N2O[1:]-D_N2O[:-1]),color='#FFCC00',lw=2,label='d_N2O')
    plt.title('N2O fluxes (MtN/yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # lifetime
    ax = plt.subplot(2,3,3)
    plt.plot(1700+np.arange(ind_final+1),alpha_N2O*(N2O_0+D_N2O)/(alpha_N2O*N2O_0/tau_N2O_hv-D_HVSNK_N2O),color='k',lw=2,label='OSCAR')
    plt.title('N2O lifetime (yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

#========
# 2.4. O3
#========

def plot_O3(D_O3t,D_O3s,D_EESC,D_N2O_lag,D_gst):
    plt.figure()

    # tropospheric O3
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_O3t,color='k',lw=2,label='OSCAR')
    plt.title('$\Delta$O3 trop. (DU)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # stratospheric O3
    ax = plt.subplot(2,3,2)
    plt.plot(1700+np.arange(ind_final+1),D_O3s,color='k',lw=2,label='OSCAR')
    plt.title('$\Delta$O3 strat. (DU)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # EESC
    ax = plt.subplot(2,3,3)
    plt.plot(1700+np.arange(ind_final+1),D_EESC,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(ind_final+1),(chi_O3s_N2O*D_N2O_lag*(1-D_EESC/EESC_x)/chi_O3s_EESC),color='k',lw=1,label='N2O effect')    
    plt.title('$\Delta$EESC (ppt)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # age-of-air
    ax = plt.subplot(2,3,4)
    plt.plot(1700+np.arange(ind_final+1),tau_lag/(1+gamma_age*D_gst),color='k',lw=2,label='OSCAR')
    plt.title('mean age-of-air (yr)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])    

#==============
# 2.5. Aerosols
#==============

def plot_AER(D_SO4,D_POA,D_BC,D_NO3,D_SOA,D_AERh,RF_SO4,RF_POA,RF_BC,RF_NO3,RF_SOA,RF_cloud):
    plt.figure()

    # atmospheric burden
    ax = plt.subplot(2,3,1)
    plt.plot(1700+np.arange(ind_final+1),D_SO4,color='b',lw=2,label='D_SO4')
    plt.plot(1700+np.arange(ind_final+1),D_POA,color='m',lw=2,label='D_POA')
    plt.plot(1700+np.arange(ind_final+1),D_BC,color='r',lw=2,label='D_BC')
    plt.plot(1700+np.arange(ind_final+1),D_NO3,color='g',lw=2,label='D_NO3')
    plt.plot(1700+np.arange(ind_final+1),D_SOA,color='y',lw=2,label='D_SOA')    
    plt.plot(1700+np.arange(ind_final+1),D_AERh,color='c',lw=2,label='D_AERh') 
    plt.title('$\Delta$ burdens (Tg)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # radiative forcing
    ax = plt.subplot(2,3,4)
    plt.plot([1700,1700+ind_final],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),RF_SO4,color='b',lw=2,label='RF_SO4')
    plt.errorbar([2010],[-0.40],yerr=[[0.20],[0.20]],marker='o',mfc='b',color='k')
    plt.plot(1700+np.arange(ind_final+1),RF_POA,color='m',lw=2,label='RF_POA')
    plt.errorbar([2010],[-0.29],yerr=[[-0.29*0.63],[-0.29*0.72]],marker='o',mfc='m',color='k')
    plt.plot(1700+np.arange(ind_final+1),RF_BC,color='r',lw=2,label='RF_BC')
    plt.errorbar([2010],[+0.60],yerr=[[+0.60*0.61],[+0.60*0.70]],marker='o',mfc='r',color='k')
    plt.plot(1700+np.arange(ind_final+1),RF_NO3,color='g',lw=2,label='RF_NO3')
    plt.errorbar([2010],[-0.11],yerr=[[0.19],[0.08]],marker='o',mfc='g',color='k')
    plt.plot(1700+np.arange(ind_final+1),RF_SOA,color='y',lw=2,label='RF_SOA')
    plt.errorbar([2010],[-0.03],yerr=[[0.24],[0.23]],marker='o',mfc='y',color='k') 
    plt.plot(1700+np.arange(ind_final+1),RF_cloud,color='c',lw=2,label='RF_cloud') 
    plt.errorbar([2010],[-0.45],yerr=[[0.75],[0.45]],marker='o',mfc='c',color='k')
    #plt.errorbar([2010],[-0.10],yerr=[[0.20],[0.20]],marker='o',mfc='0.5',color='k')
    plt.title('RF (W/m2)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,max(1700+ind_final,2010+10)])

#=============
# 2.6. Climate
#=============

def plot_clim(RF,D_gst,D_gyp,RF_CO2,RF_CH4,RF_H2Os,RF_N2O,RF_halo,RF_O3t,RF_O3s,RF_SO4,RF_POA,RF_BC,RF_NO3,RF_SOA,RF_cloud,RF_BCsnow,RF_LCC,RFcon,RFvolc,RFsolar):
    plt.figure()

    # radiative forcing
    ax = plt.subplot(2,3,1)
    plt.plot([1700,1700+ind_final],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),RF,color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(RF_ipcc)),RF_ipcc,color='r',lw=2,ls='--',label='IPCC')
    plt.title('RF (W/m2)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # global temperature
    ax = plt.subplot(2,3,2)
    plt.plot([1700,1700+ind_final],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),D_gst-np.mean(D_gst[200:230]),color='k',lw=2,label='OSCAR')
    plt.plot(1700+np.arange(len(gst_giss)),gst_giss-np.mean(gst_giss[200:230]),color='b',ls='--',label='GISS')
    plt.plot(1700+np.arange(len(gst_had)),gst_had-np.mean(gst_had[200:230]),color='g',ls='--',label='Hadley')
    plt.plot(1700+np.arange(len(gst_ncdc)),gst_ncdc-np.mean(gst_ncdc[200:230]),color='m',ls='--',label='NCDC')
    plt.title('$\Delta$ temp. (K)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # global precipitations
    ax = plt.subplot(2,3,3)
    plt.plot(1700+np.arange(ind_final+1),D_gyp,color='k',lw=2,label='OSCAR')
    plt.title('$\Delta$ precip. (mm)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])

    # RF details
    ax = plt.subplot(2,3,4)
    plt.plot([1700,1700+ind_final],[0,0],'k-')
    plt.plot(1700+np.arange(ind_final+1),RF_CO2+RF_CH4+RF_N2O+RF_halo+RF_H2Os,color='r',lw=2,label='WMGHG')
    plt.plot(1700+np.arange(len(RF_WMGHG_ipcc)),RF_WMGHG_ipcc,color='r',ls='--')
    plt.plot(1700+np.arange(ind_final+1),RF_O3t+RF_O3s,color='y',lw=2,label='O3')
    plt.plot(1700+np.arange(len(RF_O3_ipcc)),RF_O3_ipcc,color='y',ls='--')    
    plt.plot(1700+np.arange(ind_final+1),RF_SO4+RF_POA+RF_BC+RF_NO3+RF_SOA+RF_cloud,color='b',lw=2,label='AER')
    plt.plot(1700+np.arange(len(RF_AER_ipcc)),RF_AER_ipcc,color='b',ls='--')
    plt.plot(1700+np.arange(ind_final+1),RF_BCsnow+RF_LCC,color='g',lw=2,label='Alb.')
    plt.plot(1700+np.arange(len(RF_Alb_ipcc)),RF_Alb_ipcc,color='g',ls='--')
    plt.plot(1700+np.arange(ind_final+1),RFcon,color='k',ls='--',label='Ant.')
    plt.plot(1700+np.arange(ind_final+1),RFvolc+RFsolar,color='0.5',ls='--',label='Nat.')
    plt.title('RF (W/m2)',fontsize='medium')
    plt.legend(loc=0,ncol=2,prop=FontProperties(size='small'))
    plt.xticks(rotation=27)
    ax.set_xlim([1700,1700+ind_final])


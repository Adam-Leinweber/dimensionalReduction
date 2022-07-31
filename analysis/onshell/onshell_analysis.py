import numpy as np

def calc_3lep_onshell(N_leps, electronE, electronPT, electronEta, electronPhi, muonE, muonPT, muonEta, muonPhi, jetPT, jetEta, jetPhi, MET, METPhi, btag, electronCharge, muonCharge):
    # input: lepton information
    # output: True if event passes selection criteria, False if event does not
    # Simply require 3 leptons and a lepton PT cut

    valid = True
    
    sfos = False
    mllsfos = -np.inf
    m3l_mz = -np.inf
    HT = -1
    mT = -np.inf
    deltaRsfos = np.inf
    deltaR3l = np.inf
    
    N_jets = len(jetPT)
    
    #BASELINE REQUIREMENTS
    #Remove leptons/jets with pT < 12/20 GeV and outside eta bounds
    for i in range(len(electronPT)):
        if electronPT[i] < 18:
            #electronPT[i] = 0
            N_leps -= 1
        elif abs(electronEta[i]) > 2.47:
            #electronPT[i] = 0
            N_leps -= 1
    for i in range(len(muonPT)):
        if muonPT[i] < 14.7:
            #muonPT[i] = 0
            N_leps -= 1
        elif abs(muonEta[i]) > 2.5:
            #muonPT[i] = 0
            N_leps -= 1
    for i in range(len(jetPT)):
        if jetPT[i] < 20:
            #jetPT[i] = 0
            N_jets -= 1
        elif abs(jetEta[i]) > 4.5:
            #jetPT[i] = 0
            N_jets -= 1


    # If we have enough leptons, check their PT is high enough
    if N_leps == 3:
        leptonPT = np.sort(np.concatenate((electronPT, muonPT), axis = None))[::-1]
        if leptonPT[0] < 25:
            N_leps -= 1
        elif leptonPT[1] < 20:
            N_leps -= 1
        elif leptonPT[2] < 10:
            N_leps -= 1

    #SIGNAL REQUIREMENTS
    # nSFOS >= 1
    if abs(sum(electronCharge)) == len(electronCharge):
        if abs(sum(muonCharge)) == len(muonCharge):
            valid = False
        else:
            sfos = True
    elif abs(sum(muonCharge)) == len(muonCharge):
        if abs(sum(electronCharge)) == len(electronCharge):
            valid = False
        else:
            sfos = True
    else:
        sfos = True
        
    # 3 Lepton cut
    if N_leps != 3 or sfos == False:
        # If don't have enough leptons valid = False
        sfos = False
        valid = False
        return valid, mllsfos, m3l_mz, HT, mT, deltaRsfos, deltaR3l, sfos


    # no b-jets
    if sum(btag) != 0:
        valid = False  
    
    # MET cut
    if MET < 50:
        valid = False
                
    # mll is calculated with pair closest to Z mass
    # mllsfos in [75, 105]
    mz = 91.1876
    if sum(electronCharge) == 0 and len(muonCharge) == 1: #e sfos, mu
        mllsfos = np.sqrt((electronE[0] + electronE[1])**2 - (electronPT[0] + electronPT[1])**2)
        mT = np.sqrt(2*muonPT[0]*MET*(1-np.cos(muonPhi[0] - METPhi)))
    elif sum(muonCharge) == 0 and len(electronCharge) == 1: # mu sfos, e
        mllsfos =  np.sqrt((muonE[0] + muonE[1])**2 - (muonPT[0] + muonPT[1])**2)
        mT = np.sqrt(2*electronPT[0]*MET*(1-np.cos(electronPhi[0] - METPhi)))
    elif sum(electronCharge) != 0 and len(muonCharge) == 0: # 3 e
        pos_e_E = electronE[electronCharge == +1]
        neg_e_E = electronE[electronCharge == -1]
        pos_e_pT = electronPT[electronCharge == +1]
        neg_e_pT = electronPT[electronCharge == -1]
        select_We_pT = electronPT[electronCharge == sum(electronCharge)]
        select_We_phi = electronPhi[electronCharge == sum(electronCharge)]
        for i in range(len(pos_e_E)):
            for j in range(len(neg_e_E)):
                if abs(np.sqrt((pos_e_E[i] + neg_e_E[j])**2 - (pos_e_pT[i] + neg_e_pT[j])**2) - mz) < abs(mllsfos - mz):
                    mllsfos =  np.sqrt((pos_e_E[i] + neg_e_E[j])**2 - (pos_e_pT[i] + neg_e_pT[j])**2)
                    if sum(electronCharge) > 0:
                        if i == 0:
                            index = 1
                        if i == 1:
                            index = 0
                        mT = np.sqrt(2*select_We_pT[index]*MET*(1-np.cos(select_We_phi[index] - METPhi)))
                    elif sum(electronCharge) < 0:
                        if j == 0:
                            index = 1
                        if j == 1:
                            index = 0
                        mT = np.sqrt(2*select_We_pT[index]*MET*(1-np.cos(select_We_phi[index] - METPhi)))
    elif sum(muonCharge) != 0 and len(electronCharge) == 0: # 3 mu
        pos_m_E = muonE[muonCharge == +1]
        neg_m_E = muonE[muonCharge == -1]
        pos_m_pT = muonPT[muonCharge == +1]
        neg_m_pT = muonPT[muonCharge == -1]
        select_Wm_pT = muonPT[muonCharge == sum(muonCharge)]
        select_Wm_phi = muonPhi[muonCharge == sum(muonCharge)]
        for i in range(len(pos_m_E)):
            for j in range(len(neg_m_E)):
                if abs(np.sqrt((pos_m_E[i] + neg_m_E[j])**2 - (pos_m_pT[i] + neg_m_pT[j])**2) - mz) < abs(mllsfos - mz):
                    mllsfos =  np.sqrt((pos_m_E[i] + neg_m_E[j])**2 - (pos_m_pT[i] + neg_m_pT[j])**2)
                    if sum(muonCharge) > 0:
                        if i == 0:
                            index = 1
                        if i == 1:
                            index = 0
                        mT = np.sqrt(2*select_Wm_pT[index]*MET*(1-np.cos(select_Wm_phi[index] - METPhi)))
                    elif sum(muonCharge) < 0:
                        if j == 0:
                            index = 1
                        if j == 1:
                            index = 0
                        mT = np.sqrt(2*select_Wm_pT[index]*MET*(1-np.cos(select_Wm_phi[index] - METPhi)))
    if mllsfos > 75:
        valid = False
    if mT < 50:
        valid = False
                
    # Trilepton invariant mass must be off the Z boson mass 
    E = np.concatenate((electronE, muonE))
    PT = np.concatenate((electronPT, muonPT))
    m3l = np.sqrt((sum(E))**2 - (sum(PT))**2)
    mz = 91.1876
    m3l_mz = abs(m3l - mz) 
    if m3l_mz < 15:
        valid = False

    # HT = sum of jetpt, maybe use for further analysis
    HT = sum(jetPT)
    if HT > 200 or HT == 0:
        valid = False
    
    # # mT defined for lepton from W (non sfos pair)
    # mw = 80.379
    # if sum(electronCharge) == 0 and len(muonCharge) == 1: #e sfos, mu
    #     mT = np.sqrt(2*muonPT[0]*MET*(1-np.cos(muonPhi[0] - METPhi)))
    # elif sum(muonCharge) == 0 and len(electronCharge) == 1: # mu sfos, e
    #     mT = np.sqrt(2*electronPT[0]*MET*(1-np.cos(electronPhi[0] - METPhi)))
    # elif sum(electronCharge) != 0 and len(muonCharge) == 0: # 3 e
    #     select_e_pT = electronEta[electronCharge == sum(electronCharge)]
    #     select_e_phi = electronPhi[electronCharge == sum(electronCharge)]
    #     for i in range(len(select_e_pT)):
    #         if abs(np.sqrt(2*select_e_pT[i]*MET*(1-np.cos(select_e_phi[i] - METPhi))) - mw) < abs(mT - mw):
    #             mT = np.sqrt(2*select_e_pT[i]*MET*(1-np.cos(select_e_phi[i] - METPhi)))
    # elif sum(muonCharge) != 0 and len(electronCharge) == 0: # 3 mu
    #     select_m_pT = muonEta[muonCharge == sum(muonCharge)]
    #     select_m_phi = muonPhi[muonCharge == sum(muonCharge)]
    #     for i in range(len(select_m_pT)):
    #         if abs(np.sqrt(2*select_m_pT[i]*MET*(1-np.cos(select_m_phi[i] - METPhi))) - mw) < abs(mllsfos - mw):
    #             mT = np.sqrt(2*select_m_pT[i]*MET*(1-np.cos(select_m_phi[i] - METPhi)))

    
    # min delta R
    if N_leps > 2:
        phi_angles = np.concatenate((electronPhi, muonPhi))
        eta_angles = np.concatenate((electronEta, muonEta))
        for i in range(len(phi_angles)):
            for j in range(i+1, len(phi_angles-1)):
                deltaR3l_temp = np.sqrt((eta_angles[i] - eta_angles[j])**2 + (phi_angles[i] - phi_angles[j])**2)
                if deltaR3l_temp < deltaR3l:
                    deltaR3l = deltaR3l_temp

    # min delta Rsfos
    if sum(electronCharge) == 0 and len(muonCharge) == 1: #e sfos, mu
        deltaRsfos = np.sqrt((electronPhi[0] - electronPhi[1])**2 + (electronEta[0] - electronEta[1])**2)
    elif sum(muonCharge == 0) and len(electronCharge) == 1: # mu sfos, e
        deltaRsfos =  np.sqrt((muonPhi[0] - muonPhi[1])**2 + (muonEta[0] - muonEta[1])**2)
    elif sum(electronCharge) != 0 and len(muonCharge) == 0: # 3 e
        pos_e_phi = electronPhi[electronCharge == +1]
        neg_e_phi = electronPhi[electronCharge == -1]
        pos_e_eta = electronEta[electronCharge == +1]
        neg_e_eta = electronEta[electronCharge == -1]
        for i in range(len(pos_e_phi)):
            for j in range(len(neg_e_phi)):
                if np.sqrt((pos_e_phi[i] - neg_e_phi[j])**2 + (pos_e_eta[i] - neg_e_eta[j])**2) < deltaRsfos:
                    deltaRsfos =  np.sqrt((pos_e_phi[i] - neg_e_phi[j])**2 + (pos_e_eta[i] - neg_e_eta[j])**2)
    elif sum(muonCharge) != 0 and len(electronCharge) == 0: # 3 mu
        pos_m_phi = muonPhi[muonCharge == +1]
        neg_m_phi = muonPhi[muonCharge == -1]
        pos_m_eta = muonEta[muonCharge == +1]
        neg_m_eta = muonEta[muonCharge == -1]
        for i in range(len(pos_m_phi)):
            for j in range(len(neg_m_phi)):
                if np.sqrt((pos_m_phi[i] - neg_m_phi[j])**2 + (pos_m_eta[i] - neg_m_eta[j])**2) < deltaRsfos:
                    deltaRsfos = np.sqrt((pos_m_phi[i] - neg_m_phi[j])**2 + (pos_m_eta[i] - neg_m_eta[j])**2)


    # if variables aren't defined for some reason
    if mllsfos == -np.inf:
        sfos = False
    if m3l_mz == -np.inf:
        sfos = False
    if mT == -np.inf:
        sfos = False
    if deltaRsfos == np.inf:
        sfos = False
    if deltaR3l == np.inf:
        sfos = False
        
    return valid, mllsfos, m3l_mz, HT, mT, deltaRsfos, deltaR3l, sfos

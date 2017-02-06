from ROOT import *
import numpy as np

myfile = TFile("../data/ntuple_v3_2000k.root")
mytree = myfile.Get("SimpleJet") 

clusters=[]
cells=[]
nparticles=[]

# see README.md file
# https://gitlab.cern.ch/mleblanc/ClusterSplitting/blob/master/README.md

for i in range(mytree.GetEntries()):
    mytree.GetEntry(i)
    if (i%100==0):
        print i

    # jet = (pT, eta, phi, PM) 
    # pT : transverse momentum
    # PM : mass of the truth jets
    # why (1,2,3,4)?
    leadjet = TLorentzVector(1,2,3,4)

    # iterate over number of truth jets
    # leadjet is the jet with highest pT
    # (The leading two such jets should always have mass near 80 or 90 GeV)
    for j in range(mytree.NJets):
        if (mytree.JetsPt[j] > leadjet.Pt()):
            leadjet.SetPtEtaPhiM(mytree.JetsPt[j],mytree.JetsEta[j],mytree.JetsPhi[j],mytree.JetsPM[j])
            pass
        pass

    # iterate over topocluster
    # energy is in GeV
    for j in range(len(mytree.Topocluster_E)):
        topovec = TLorentzVector()
        topovec.SetPtEtaPhiM(mytree.Topocluster_E[j]/np.cosh(mytree.Topocluster_eta[j]),mytree.Topocluster_eta[j],mytree.Topocluster_phi[j],0.)
        #only take the clusters inside the highest pT jet!
        if (leadjet.DeltaR(topovec) > 0.4): 
            continue
        clusters+=[topovec.Pt()]
        cluster_cells=[]    
        for k in range(len(mytree.Topocluster_cellIDs[j])):
            k2 = mytree.Topocluster_cellIDs[j][k]
            cellvector = TLorentzVector()
            # why 0.001 to make it in GeV
            # weight in topocluster -> topocluster algo
            cellvector.SetPtEtaPhiM(0.001*mytree.Topocluster_cellWeights[j][k]*mytree.Cell_E[k2]/np.cosh(mytree.Cell_eta[k2]),mytree.Cell_eta[k2],mytree.Cell_phi[k2],0.)
            cluster_cells+=[cellvector.Pt()]
        pass
        cells+=[cluster_cells]
        nparts=0
        for k in range(len(mytree.Topocluster_truthEfrac[j])):
            if (mytree.Topocluster_truthEfrac[j][k] > 0.1):
                nparts+=1
                pass
            pass
        nparticles+=[nparts]   
    break
  
print clusters
print cells
print nparticles
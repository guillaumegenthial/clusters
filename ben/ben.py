import ROOT
import numpy as np

# 0. ROOT data
myfile = ROOT.TFile("../data/ntuple_v3_2000k.root")
mytree = myfile.Get("SimpleJet")

canvas = ROOT.TCanvas('draw', 'draw', 0, 0, 800, 800)
canvas.cd()
Nfrac = ROOT.TH1F("Nfrac", "Nfrac", 1000, -0.01, 0.1)

# myfile.Print()

# 2. iterate over entries
for i in range(min(mytree.GetEntries(), 2000)):
    mytree.GetEntry(i)
    if (i%100==0):
        print i

    # print len(mytree.Cell_barcodes) # print 0

    leadjet = ROOT.TLorentzVector(1,2,3,4)
    for j in range(mytree.NJets):
        if (mytree.JetsPt[j] > leadjet.Pt()):
            leadjet.SetPtEtaPhiM(mytree.JetsPt[j],mytree.JetsEta[j],mytree.JetsPhi[j],mytree.JetsPM[j])
            pass
        pass

    for j in range(len(mytree.Topocluster_E)):
        topovec = ROOT.TLorentzVector()
        topovec.SetPtEtaPhiM(mytree.Topocluster_E[j]/np.cosh(mytree.Topocluster_eta[j]),mytree.Topocluster_eta[j],mytree.Topocluster_phi[j],0.)
        #only take the clusters inside the highest pT jet!
        if (leadjet.DeltaR(topovec) > 0.4): 
            continue
        
        for k in range(len(mytree.Topocluster_truthEfrac[j])):
            nparts = 0
            prop = mytree.Topocluster_truthEfrac[j][k]
            Nfrac.Fill(prop)
            if (prop > 0.1):
                nparts+=1
                pass
            pass
  
    
Nfrac.Draw('hist')
Nfrac.SetMaximum(Nfrac.GetMaximum()*1.5)
Nfrac.GetXaxis().SetTitle("Distribution of energy contribution to cluster")
canvas.SaveAs("Nfrac.pdf")


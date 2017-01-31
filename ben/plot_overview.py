import ROOT
from ROOT import gROOT,gPad,gStyle,TCanvas,TFile,TLine,TLatex,TAxis,TLegend,TPostScript
import math

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(False)
ROOT.gStyle.SetOptTitle(False)
ROOT.gStyle.SetLegendBorderSize(0);
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)

legend=TLegend(0.15,0.5,0.35,0.6)
legend2=TLegend(0.65,0.5,0.85,0.6)

l=TLatex()
l.SetNDC()
l.SetTextFont(72)
p=TLatex()
p.SetNDC();
p.SetTextFont(42)
q=TLatex()
q.SetNDC();
q.SetTextFont(42)
q.SetTextSize(0.03)
r=TLatex()
r.SetNDC();
r.SetTextFont(42)
r.SetTextSize(0.03)
s=TLatex()
s.SetNDC();
s.SetTextFont(42)
s.SetTextSize(0.03)

canvas = ROOT.TCanvas('draw', 'draw', 0, 0, 800, 800)
pad = ROOT.TPad()
canvas.cd()

infile = ROOT.TFile('../data/ntuple_v2_500.root','open')
intree = infile.Get('SimpleJet')

# intree.Draw('mu>>htemp')
# htemp = ROOT.gDirectory.Get("htemp")

# htemp.Scale(1.0/htemp.Integral())

# htemp.GetXaxis().SetTitle('< #mu >')
# htemp.GetYaxis().SetTitle('Fraction of events')
# htemp.GetXaxis().SetTitleOffset(0.95)
# htemp.GetYaxis().SetTitleOffset(1.0)
# htemp.GetXaxis().SetLabelSize(0.02)
# htemp.GetYaxis().SetLabelSize(0.02)
# htemp.SetMaximum(1.25*htemp.GetMaximum())

# htemp.Draw('hist')

# l.DrawLatex(0.15,0.8,"ATLAS")
# p.DrawLatex(0.32,0.8,'Internal Simulation')
# q.DrawLatex(0.15,0.75,'DSID 410000 (t#bar{t})')

# canvas.SaveAs('mu.pdf')

h_NJets = ROOT.TH1F('NJets','NJets',20,0.5,19.5)
h_JetsPt = ROOT.TH1F('JetsPt','JetsPt',50,0.5,2999.5)
h_JetsEta = ROOT.TH1F('JetsEta','JetsEta',35,-3.5,3.5)
h_JetsPhi = ROOT.TH1F('JetsPhi','JetsPhi',32,-3.2,3.2)
h_JetsM = ROOT.TH1F('JetsM','JetsM',60,0.5,149.5)

h_NJets_EM = ROOT.TH1F('NJets_EM','NJets_EM',20,0.5,19.5)
h_Jets_EM_Pt = ROOT.TH1F('Jets_EM_Pt','Jets_EM_Pt',50,0.5,2999.5)
h_Jets_EM_Eta = ROOT.TH1F('Jets_EM_Eta','Jets_EM_Eta',35,-3.5,3.5)
h_Jets_EM_Phi = ROOT.TH1F('Jets_EM_Phi','Jets_EM_Phi',32,-3.2,3.2)
h_Jets_EM_M = ROOT.TH1F('Jets_EM_M','Jets_EM_M',60,0.5,149.5)

h_NJets_LC = ROOT.TH1F('NJets_LC','NJets_LC',20,0.5,19.5)
h_Jets_LC_Pt = ROOT.TH1F('Jets_LC_Pt','Jets_LC_Pt',50,0.5,2999.5)
h_Jets_LC_Eta = ROOT.TH1F('Jets_LC_Eta','Jets_LC_Eta',35,-3.5,3.5)
h_Jets_LC_Phi = ROOT.TH1F('Jets_LC_Phi','Jets_LC_Phi',32,-3.2,3.2)
h_Jets_LC_M = ROOT.TH1F('Jets_LC_M','Jets_LC_M',60,0.5,149.5)

h_NJets_PF = ROOT.TH1F('NJets_PF','NJets_PF',20,0.5,19.5)
h_Jets_PF_Pt = ROOT.TH1F('Jets_PF_Pt','Jets_PF_Pt',50,0.5,2999.5)
h_Jets_PF_Eta = ROOT.TH1F('Jets_PF_Eta','Jets_PF_Eta',35,-3.5,3.5)
h_Jets_PF_Phi = ROOT.TH1F('Jets_PF_Phi','Jets_PF_Phi',32,-3.2,3.2)
h_Jets_PF_M = ROOT.TH1F('Jets_PF_M','Jets_PF_M',60,0.5,149.5)

h_NTopoClusters = ROOT.TH1F('NTopoClusters','NTopoClusters',500,0.5,499.5)
h_TopoClustersEta = ROOT.TH1F('TopoClustersEta','TopoClustersEta',35,-3.5,3.5)
h_TopoClustersPhi = ROOT.TH1F('TopoClustersPhi','TopoClustersPhi',32,-3.2,3.2)
h_TopoClustersE = ROOT.TH1F('TopoClustersE','TopoClustersE',40,0.5,399.5)

h_NTrack = ROOT.TH1F('NTrack','NTrack',500,0.5,499.5)
h_TrackPt = ROOT.TH1F('TrackPt','TrackPt',40,0.5,3999.5)
h_TrackEta = ROOT.TH1F('TrackEta','TrackEta',35,-3.5,3.5)
h_TrackPhi = ROOT.TH1F('TrackPhi','TrackPhi',32,-3.2,3.2)

h_response_m_EM = ROOT.TH1F('response_m_EM','response_m_EM',50,0,2)
h_response_m_LC = ROOT.TH1F('response_m_LC','response_m_LC',50,0,2)
h_response_m_PF = ROOT.TH1F('response_m_PF','response_m_PF',50,0,2)

for entry in range(0,intree.GetEntries()):
	intree.GetEntry(entry)

	if entry%1000==0 : 
		print 'entry '+str(entry)+'/'+str(intree.GetEntries())
		
	h_NJets.Fill(intree.NJets, intree.TheWeight)
	tlv_truthjets = []
	for jet in range(0, intree.NJets):
		tlv = ROOT.TLorentzVector()
		tlv.Clear()
		tlv.SetPtEtaPhiM(intree.JetsPt[jet],
						 intree.JetsEta[jet],
						 intree.JetsPhi[jet],
						 intree.JetsPM[jet])
		tlv_truthjets.append(tlv)

		if(tlv.Pt() > 500.):
			# h_JetsPt.Fill(intree.JetsPt[jet],intree.TheWeight)
			# h_JetsEta.Fill(intree.JetsEta[jet],intree.TheWeight)	
			# h_JetsPhi.Fill(intree.JetsPhi[jet],intree.TheWeight)
			# h_JetsM.Fill(intree.JetsPM[jet],intree.TheWeight)
			h_JetsPt.Fill(tlv.Pt(),intree.TheWeight)
			h_JetsEta.Fill(tlv.Eta(),intree.TheWeight)
			h_JetsPhi.Fill(tlv.Phi(),intree.TheWeight)
			h_JetsM.Fill(tlv.M(),intree.TheWeight)

	h_NJets_EM.Fill(intree.JetsPt_EM.size(), intree.TheWeight)
	tlv_emjets = []
	for jet in range(0, intree.JetsPt_EM.size()):
		tlv = ROOT.TLorentzVector()
		tlv.Clear()
		tlv.SetPtEtaPhiM(intree.JetsPt_EM[jet],
						 intree.JetsEta_EM[jet],
						 intree.JetsPhi_EM[jet],
						 intree.JetsPM_EM[jet])
		tlv_emjets.append(tlv)

		if(tlv.Pt() > 500.):
			h_Jets_EM_Pt.Fill(tlv.Pt(),intree.TheWeight)
			h_Jets_EM_Eta.Fill(tlv.Eta(),intree.TheWeight)
			h_Jets_EM_Phi.Fill(tlv.Phi(),intree.TheWeight)
			h_Jets_EM_M.Fill(tlv.M(),intree.TheWeight)

	h_NJets_LC.Fill(intree.JetsPt_LC.size(), intree.TheWeight)
	tlv_lcjets = []
	for jet in range(0, intree.JetsPt_LC.size()):
		tlv = ROOT.TLorentzVector()
		tlv.Clear()
		tlv.SetPtEtaPhiM(intree.JetsPt_LC[jet],
						 intree.JetsEta_LC[jet],
						 intree.JetsPhi_LC[jet],
						 intree.JetsPM_LC[jet])
		tlv_lcjets.append(tlv)

		if(tlv.Pt() > 500.):
			h_Jets_LC_Pt.Fill(tlv.Pt(),intree.TheWeight)
			h_Jets_LC_Eta.Fill(tlv.Eta(),intree.TheWeight)
			h_Jets_LC_Phi.Fill(tlv.Phi(),intree.TheWeight)
			h_Jets_LC_M.Fill(tlv.M(),intree.TheWeight)

	h_NJets_PF.Fill(intree.JetsPt_PF.size(), intree.TheWeight)
	tlv_pfjets = []
	for jet in range(0, intree.JetsPt_PF.size()):
		tlv = ROOT.TLorentzVector()
		tlv.Clear()
		tlv.SetPtEtaPhiM(intree.JetsPt_PF[jet],
						 intree.JetsEta_PF[jet],
						 intree.JetsPhi_PF[jet],
						 intree.JetsPM_PF[jet])
		tlv_pfjets.append(tlv)

		if(tlv.Pt() > 500.):
			h_Jets_PF_Pt.Fill(tlv.Pt(),intree.TheWeight)
			h_Jets_PF_Eta.Fill(tlv.Eta(),intree.TheWeight)
			h_Jets_PF_Phi.Fill(tlv.Phi(),intree.TheWeight)
			h_Jets_PF_M.Fill(tlv.M(),intree.TheWeight)

	h_NTopoClusters.Fill(intree.Topocluster_N, intree.TheWeight)
	tlv_clusters = []
	for cluster in range(0, intree.Topocluster_N):
		tlv = ROOT.TLorentzVector()
		tlv.Clear()
		tlv.SetPtEtaPhiM(intree.Topocluster_E[cluster]/math.cosh(intree.Topocluster_eta[cluster]),
						 intree.Topocluster_eta[cluster],
						 intree.Topocluster_phi[cluster],
						 0)
		tlv_clusters.append(tlv)
		# Fill basic histograms
		h_TopoClustersEta.Fill(intree.Topocluster_eta[cluster],intree.TheWeight)	
		h_TopoClustersPhi.Fill(intree.Topocluster_phi[cluster],intree.TheWeight)
		h_TopoClustersE.Fill(intree.Topocluster_E[cluster],intree.TheWeight)
		#h_NTopoClustersPerJet.Fill(intree.Topocluster_N, intree.TheWeight)

	h_NTrack.Fill(intree.Track_N, intree.TheWeight)
	tlv_tracks = []
	for track in range(0, intree.Track_N):
		tlv = ROOT.TLorentzVector()
		tlv.Clear()
		tlv.SetPtEtaPhiM(intree.Track_Pt[track],
						 intree.Track_Eta[track],
						 intree.Track_Phi[track],
						 0)
		tlv_tracks.append(tlv)
		# Fill basic histograms
		h_TrackPt.Fill(intree.Track_Pt[track],intree.TheWeight)
		h_TrackEta.Fill(intree.Track_Eta[track],intree.TheWeight)	
		h_TrackPhi.Fill(intree.Track_Phi[track],intree.TheWeight)
		#h_NTrackPerJet.Fill(intree.Topocluster_N, intree.TheWeight)

	# ## Loop over jet TLVs
	# for truthjet in tlv_truthjets:
	# 	tlv_cluster_sum = ROOT.TLorentzVector()
	# 	tlv_cluster_sum.Clear()
	# 	for cluster in tlv_clusters:
	# 		if(truthjet.DeltaR(cluster) < 0.4):
	# 			tlv_cluster_sum+=cluster

	for truthjet in tlv_truthjets:
		for emjet in tlv_emjets:
			if(truthjet.DeltaR(emjet)<0.4 and truthjet.M()>0.1 and abs(truthjet.Eta()) < 0.4 and abs(emjet.Eta()) < 0.4):
				h_response_m_EM.Fill(emjet.M()/truthjet.M())
		for lcjet in tlv_lcjets:
			if(truthjet.DeltaR(lcjet)<0.4 and truthjet.M()>0.1 and abs(truthjet.Eta()) < 0.4 and abs(emjet.Eta()) < 0.4):
				h_response_m_LC.Fill(lcjet.M()/truthjet.M())
		for pfjet in tlv_pfjets:
			if(truthjet.DeltaR(pfjet)<0.4 and truthjet.M()>0.1 and abs(truthjet.Eta()) < 0.4 and abs(emjet.Eta()) < 0.4):
				h_response_m_PF.Fill(pfjet.M()/truthjet.M())

### END LOOP OVER TREE ENTRIES !!!

### Basic truth jet plots

h_NJets.Draw('hist')
h_NJets.SetMaximum(h_NJets.GetMaximum()*1.5)
h_NJets.GetXaxis().SetTitle('Truth jet multiplicity')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
canvas.SaveAs('plots/NJets.pdf')

h_JetsPt.Draw('hist')
h_JetsPt.SetMaximum(h_JetsPt.GetMaximum()*1.25)
h_JetsPt.GetXaxis().SetTitle('Truth jet p_{T} [GeV]')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")

canvas.SetLogy()
canvas.RedrawAxis()
canvas.SaveAs('plots/JetsPt.pdf')
canvas.SetLogy(0)

h_JetsEta.Draw('hist')
h_JetsEta.SetMaximum(h_JetsEta.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_JetsEta.GetXaxis().SetTitle('Truth jet #eta')
canvas.SaveAs('plots/JetsEta.pdf')

h_JetsPhi.Draw('hist')
h_JetsPhi.SetMaximum(h_JetsPhi.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_JetsPhi.GetXaxis().SetTitle('Truth jet #phi')
canvas.SaveAs('plots/JetsPhi.pdf')

h_JetsM.Draw('hist')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_JetsM.GetXaxis().SetTitle('Truth jet mass [GeV]')
canvas.SaveAs('plots/JetsM.pdf')

###


h_NJets_EM.Draw('hist')
h_NJets_EM.SetMaximum(h_NJets_EM.GetMaximum()*1.5)
h_NJets_EM.GetXaxis().SetTitle('EM jet multiplicity')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
canvas.SaveAs('plots/NJets_EM.pdf')

h_Jets_EM_Pt.Draw('hist')
h_Jets_EM_Pt.SetMaximum(h_Jets_EM_Pt.GetMaximum()*1.25)
h_Jets_EM_Pt.GetXaxis().SetTitle('EM jet p_{T} [GeV]')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")

canvas.SetLogy()
canvas.RedrawAxis()
canvas.SaveAs('plots/Jets_EM_Pt.pdf')
canvas.SetLogy(0)

h_Jets_EM_Eta.Draw('hist')
h_Jets_EM_Eta.SetMaximum(h_Jets_EM_Eta.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_Jets_EM_Eta.GetXaxis().SetTitle('EM jet #eta')
canvas.SaveAs('plots/Jets_EM_Eta.pdf')

h_Jets_EM_Phi.Draw('hist')
h_Jets_EM_Phi.SetMaximum(h_Jets_EM_Phi.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_Jets_EM_Phi.GetXaxis().SetTitle('EM jet #phi')
canvas.SaveAs('plots/Jets_EM_Phi.pdf')

h_Jets_EM_M.Draw('hist')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_Jets_EM_M.GetXaxis().SetTitle('EM jet mass [GeV]')
canvas.SaveAs('plots/Jets_EM_M.pdf')

###

h_NJets_LC.Draw('hist')
h_NJets_LC.SetMaximum(h_NJets_LC.GetMaximum()*1.5)
h_NJets_LC.GetXaxis().SetTitle('EM jet multiplicity')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
canvas.SaveAs('plots/NJets_LC.pdf')

h_Jets_LC_Pt.Draw('hist')
h_Jets_LC_Pt.SetMaximum(h_Jets_LC_Pt.GetMaximum()*1.25)
h_Jets_LC_Pt.GetXaxis().SetTitle('EM jet p_{T} [GeV]')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")

canvas.SetLogy()
canvas.RedrawAxis()
canvas.SaveAs('plots/Jets_LC_Pt.pdf')
canvas.SetLogy(0)

h_Jets_LC_Eta.Draw('hist')
h_Jets_LC_Eta.SetMaximum(h_Jets_LC_Eta.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_Jets_LC_Eta.GetXaxis().SetTitle('EM jet #eta')
canvas.SaveAs('plots/Jets_LC_Eta.pdf')

h_Jets_LC_Phi.Draw('hist')
h_Jets_LC_Phi.SetMaximum(h_Jets_LC_Phi.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_Jets_LC_Phi.GetXaxis().SetTitle('EM jet #phi')
canvas.SaveAs('plots/Jets_LC_Phi.pdf')

h_Jets_LC_M.Draw('hist')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_Jets_LC_M.GetXaxis().SetTitle('EM jet mass [GeV]')
canvas.SaveAs('plots/Jets_LC_M.pdf')

### Topocluster plots

h_NTopoClusters.Draw('hist')
h_NTopoClusters.SetMaximum(h_NTopoClusters.GetMaximum()*1.5)
h_NTopoClusters.GetXaxis().SetTitle('Topo cluster multiplicity / event')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
canvas.SaveAs('plots/NTopoClusters.pdf')

h_TopoClustersEta.Draw('hist')
h_TopoClustersEta.SetMaximum(h_TopoClustersEta.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_TopoClustersEta.GetXaxis().SetTitle('Truth jet #eta')
canvas.SaveAs('plots/TopoClustersEta.pdf')

h_TopoClustersPhi.Draw('hist')
h_TopoClustersPhi.SetMaximum(h_TopoClustersPhi.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_TopoClustersPhi.GetXaxis().SetTitle('Truth jet #phi')
canvas.SaveAs('plots/TopoClustersPhi.pdf')

h_TopoClustersE.Draw('hist')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_TopoClustersE.GetXaxis().SetTitle('Truth jet mass [GeV]')
canvas.SaveAs('plots/TopoClustersE.pdf')

### Track plots

h_NTrack.Draw('hist')
h_NTrack.SetMaximum(h_NTrack.GetMaximum()*1.5)
h_NTrack.GetXaxis().SetTitle('Topo cluster multiplicity / event')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
canvas.SaveAs('plots/NTrack.pdf')

h_TrackPt.Draw('hist')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_TrackPt.GetXaxis().SetTitle('Track p_{T} [GeV]')
canvas.SaveAs('plots/TrackPt.pdf')

h_TrackEta.Draw('hist')
h_TrackEta.SetMaximum(h_TrackEta.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_TrackEta.GetXaxis().SetTitle('Track #eta')
canvas.SaveAs('plots/TrackEta.pdf')

h_TrackPhi.Draw('hist')
h_TrackPhi.SetMaximum(h_TrackPhi.GetMaximum()*1.5)
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
h_TrackPhi.GetXaxis().SetTitle('Track #phi')
canvas.SaveAs('plots/TrackPhi.pdf')

###
# Stack 'em

h_JetsPt.SetLineColor(2)
h_JetsPt.Draw('hist')
h_Jets_EM_Pt.SetLineColor(3)
h_Jets_EM_Pt.Draw('hist same')
h_Jets_LC_Pt.SetLineColor(4)
h_Jets_LC_Pt.Draw('hist same')
h_Jets_PF_Pt.SetLineColor(6)
h_Jets_PF_Pt.Draw('hist same')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
legend.AddEntry(h_JetsPt,'Truth jets','l')
legend.AddEntry(h_Jets_EM_Pt,'EM jets','l')
legend.AddEntry(h_Jets_LC_Pt,'LC jets','l')
legend.AddEntry(h_Jets_PF_Pt,'PFlow','l')
legend.Draw()
canvas.SaveAs('plots/stacked_JetPt.pdf')

h_JetsM.SetLineColor(2)
h_JetsM.Draw('hist')
h_Jets_EM_M.SetLineColor(3)
h_Jets_EM_M.Draw('hist same')
h_Jets_LC_M.SetLineColor(4)
h_Jets_LC_M.Draw('hist same')
h_Jets_PF_M.SetLineColor(6)
h_Jets_PF_M.Draw('hist same')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
legend.Draw()
canvas.SaveAs('plots/stacked_JetM.pdf')

# Response plots

h_response_m_EM.SetLineColor(3)
h_response_m_EM.SetMaximum(h_response_m_EM.GetMaximum()*1.75)
h_response_m_EM.Draw('hist')
h_response_m_LC.SetLineColor(4)
h_response_m_LC.Draw('hist same')
h_response_m_PF.SetLineColor(6)
h_response_m_PF.Draw('hist same')
legend2.AddEntry(h_response_m_EM,'EM jets','l')
legend2.AddEntry(h_response_m_LC,'LC jets','l')
legend2.AddEntry(h_response_m_PF,'PFlow','l')
h_Jets_PF_M.Draw('hist same')
l.DrawLatex(0.13,0.8,"ATLAS")
p.DrawLatex(0.30,0.8,"Simulation Internal")
q.DrawLatex(0.13,0.75,"Pythia8 W' #rightarrow WZ; W, Z #rightarrow qq")
s.DrawLatex(0.13,0.65,"W' mass = 5 TeV, W, Z mass = 80 GeV")
legend2.Draw()
canvas.SaveAs('plots/stacked_response_M.pdf')

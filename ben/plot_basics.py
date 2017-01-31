import ROOT
from ROOT import gROOT,gPad,gStyle,TCanvas,TFile,TLine,TLatex,TAxis,TLegend,TPostScript

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(False)
ROOT.gStyle.SetOptTitle(False)
ROOT.gStyle.SetLegendBorderSize(0);
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)

legend=TLegend(0.62,0.6,0.89,0.89)

l=TLatex()
l.SetNDC()
l.SetTextFont(72)
p=TLatex()
p.SetNDC();
p.SetTextFont(42)
q=TLatex()
q.SetNDC();
q.SetTextFont(41)
q.SetTextSize(0.03)
r=TLatex()
r.SetNDC();
r.SetTextFont(41)
r.SetTextSize(0.03)
s=TLatex()
s.SetNDC();
s.SetTextFont(41)
s.SetTextSize(0.03)

canvas = ROOT.TCanvas('draw', 'draw', 0, 0, 800, 800)
pad = ROOT.TPad()
canvas.cd()

infile = ROOT.TFile('../data/ntuple_1k.root','open')
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

for entry in range(0,intree.GetEntries()):
	intree.GetEntry(entry)

	if entry%1000==0 : 
		print 'entry '+str(entry)+'/'+str(intree.GetEntries())
		
	h_NJets.Fill(intree.NJets,intree.TheWeight)

h_NJets.Draw('hist')
canvas.SaveAs('test.pdf')

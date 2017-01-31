import ROOT

class Hist(object):
    """
    Wrapper for TH1F object
    Makes it simpler to export and plot
    """
    def __init__(self, name, title=None, nbins=20, inf=0.5, sup=19.5):
        self.name = name
        self.title = title if title is not None else name
        self.hist = ROOT.TH1F(name ,name ,nbins ,inf ,sup)

    def fill(self, x, w=None):
        if w:
            self.hist.Fill(x, w)
        else:
            self.hist.Fill(x)

    def fills(self, xs, ws=None):
        if ws is not None:
            for x, w in zip(xs, ws):
                self.fill(x, w)
        else:
            for x in xs:
                self.fill(x)

    def export(self, canvas, d="plots", suffix=""):
        self.hist.Draw('hist')
        self.hist.SetMaximum(self.hist.GetMaximum()*1.5)
        self.hist.GetXaxis().SetTitle(self.title)
        canvas.SaveAs(d+"/"+self.name+suffix+".pdf")

class Plots(object):
    """
    Container for multiple Histograms
    """
    def __init__(self):
        self.plots = dict()
        self.canvas = ROOT.TCanvas('draw', 'draw', 0, 0, 800, 800)
        self.canvas.cd()

    def add(self, name, title=None, nbins=20, inf=0.5, sup=19.5):
        self.plots[name] = Hist(name, title, nbins, inf, sup)
        return self.plots[name]

    def get(self, name):
        if name in self.plots:
            return self.plots[name]
        else:
            print "WARNING : adding new plot {}".format(name)
            return self.add(name)

    def fill(self, name, x, w=None):
        h = self.get(name)
        h.fill(x, w)

    def fills(self, name, xs, ws=None):
        h = self.get(name)
        h.fills(xs, ws)

    def export(self, d="plots", suffix=""):
        for n, h in self.plots.iteritems():
            h.export(self.canvas, d, suffix)
            

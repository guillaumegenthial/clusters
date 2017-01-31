# ClusterSplitting

Studies of track-based topological cluster splitting, from [HFSF 2016](https://indico.cern.ch/event/565930/timetable/).

Samples from Ben can be found at `/eos/atlas/user/b/bnachman/public/HFSF16/`

Newest sample: /eos/atlas/user/b/bnachman/public/HFSF16/ntuple_v3_2000k.root (2000 events)

Tree is called: SimpleJet

These are the branches that definitely work (all energies are in GeV!):
   
   ===
- NJets: Number of truth jets.   
- JetsEta: eta of the truth jets.  
- JetsPt: pT of the truth jets.  
- JetsPhi: phi of the truth jets.  
- JetsPM: mass of the truth jets.  The leading two such jets should always have mass near 80 or 90 GeV.  
   ===
- JetsEta_EM: EM-scale jet kinematics.
- JetsPt_EM
- JetsPhi_EM
- JetsPM_EM
- JetsEta_LC: LC-scale jet kinematics.
- JetsPt_LC
- JetsPhi_LC
- JetsPM_C
- JetsEta_PF: p-flow jet kinematics.
- JetsPt_PF
- JetsPhi_PF
- JetsPM_PF    
   ===   
- Topocluster_E: Topocluster energies. In GeV.
- Topocluster_eta: Topocluster eta.
- Topocluster_phi: Topocluster phi.
- Topocluster_barcodes: A vector of ints associated to each cluster.  These vectors collect the barcodes of all truth particles that contributed any energy to the cluster.  
- Topocluster_pdgids: The PDGID of the particles from Topocluster_barcodes.  
- Topocluster_truthEfrac: The fraction of the truth energy deposited in a topocluster from the particles in Topocluster_barcodes.  
- Topocluster_truthE: The energy deposited in a topocluster from the particles in Topocluster_barcodes.  
- Topocluster_SECOND_LAMBDA: This is a cluster moment (units unknown).  See [the topocluster paper](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/PERF-2014-07/).
- Topocluster_ISOLATION: This is a cluster moment. See [the topocluster paper](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/PERF-2014-07/).
- Topocluster_CENTER_LAMBDA: This is a cluster moment. See [the topocluster paper](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/PERF-2014-07/).
- Topocluster_EM_PROBABILITY: This is a cluster moment. See [the topocluster paper](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/PERF-2014-07/).
- Topocluster_SECOND_R: This is a cluster moment. See [the topocluster paper](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/PERF-2014-07/).
- Topocluster_cellN: This is the number of cells associated to this cluster.
- Topocluster_cellIDs: This is a vector per cluster that lists the index of the cell in the cell container (see below).  
- Topocluster_cellWeights: This is the weight of the cell for this topocluster.    
   ===
- Cell_N: This is the number of cells that were hit (note: energy can be negative!)  
- Cell_E: This is the energy of each cell.  This is unfortunately currently in MeV.  
- Cell_ID: This is a unique identifier for the cells.  Should actually count from 0 to Cell_N-1.  
- Cell_barcodes: This is a vector of barcodes of truth particles contributing to this cell. 
- Cell_eta: eta of the cell.
- Cell_phi: phi of the cell.
- Cell_dep: calo layer of the cell.
- Cell_vol: volume of the cell.  
   === 
- Truth_N: This is the number of truth particles is in the event.
- Truth_PDGID: This is the PDGID of the truth particles.
- Truth_barcode: This is the barcode of the truth particles.
- Truth_Pt: Self-explainatory.
- Truth_Eta: Self-explainatory.
- Truth_Phi: Self-explainatory.
- Truth_MatchTrack_ID: This is the index of the track that matches the truth particle.
- MatchTrack_MatchProb: This is the MCTruthProbability for the matched track.  Usually, we call the match kosher if this is > 0.5.      
   === 
- Track_N: This is the number of tracks (no requirements).  
- Track_Pt: Self-explainatory.  
- Track_Eta: Self-explainatory.  
- Track_Phi: Self-explainatory.  
- Track_barcode: This is the barcode of the matched truth particle.
- Track_MCprob: This is the MCTruthProbability.   Usually, we call the match kosher if this is > 0.5.  
- Track_ID: This is the index of the track.

# Wish list

- Some sweet studies.
- Brainstorm some splitting schemes
	-- Even split based on tracks
	-- Area-weighted track splitting?
	-- Energy-weighted track splitting?

# Some more information about the samples:

[JIRA](https://its.cern.ch/jira/browse/ATLMCPROD-3636)
[PANDA](https://prodtask-dev.cern.ch/prodtask/inputlist_with_request/9362/)

ESDs:

mc15_13TeV.426321.Pythia8EvtGen_A14NNPDF23LO_WprimeNarrow_WZqqqq_m80_m5000.recon.ESD.e5435_s2978_r8014_tid10116823_00_sub0352289906

IDTIDE:

/eos/atlas/user/b/bnachman/public/HFSF16/IDTIDE_BoostedW_Wprime5TeV/

# Other useful information

- Track-to-cluster association (from E/p): [github](https://github.com/jmrolsson/EoverPxAOD/blob/master/src/TrackCaloDecorator.cxx)

# Plots

- Make a plot of the topocluster response, i.e. E_cluster / (sum Topocluster_truthE).
- Make a plot of the number of truth particles required to account for X% of the topocluster's energy.  X could be 60, 70, 80.
import sys
sys.path.insert(0, '/eos/home-c/chenhua/higgsdna_finalfits_tutorial_24/HiggsDNA/higgs_dna')
from workflows.base import HggBaseProcessor
from tools.chained_quantile import ChainedQuantileRegression
from tools.SC_eta import add_photon_SC_eta
from tools.EELeak_region import veto_EEleak_flag
from tools.EcalBadCalibCrystal_events import remove_EcalBadCalibCrystal_events
from tools.gen_helpers import get_fiducial_flag, get_genJets, get_higgs_gen_attributes
from selections.photon_selections import photon_preselection
from selections.lepton_selections import select_electrons, select_muons
from selections.jet_selections import select_jets, jetvetomap
from selections.lumi_selections import select_lumis
from utils.dumping_utils import (
    diphoton_ak_array,
    dump_ak_array,
    diphoton_list_to_pandas,
    dump_pandas,
    get_obj_syst_dict,
)
from utils.misc_utils import choose_jet

from systematics import object_systematics as available_object_systematics
from systematics import object_corrections as available_object_corrections
from systematics import weight_systematics as available_weight_systematics
from systematics import weight_corrections as available_weight_corrections

import functools
import operator
import os
import warnings
from typing import Any, Dict, List, Optional
import awkward
import numpy
import sys
import vector
from coffea import processor
from coffea.analysis_tools import Weights
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)

vector.register_awkward()

def reconstruct_neutrino_pz(lepton_pt, lepton_eta, lepton_phi, met_pt, met_phi):
    """
        Reconstruct neutrino pz from W mass constraint.
    """
    MW = 80.4  # W boson mass
    # Compute lepton 4-momentum components
    lepton_px = lepton_pt * numpy.cos(lepton_phi)
    lepton_py = lepton_pt * numpy.sin(lepton_phi)
    lepton_pz = lepton_pt * numpy.sinh(lepton_eta)
    lepton_E = numpy.sqrt(lepton_px**2 + lepton_py**2 + lepton_pz**2)

    # MET components (assuming it's the neutrino)
    met_px = met_pt * numpy.cos(met_phi)
    met_py = met_pt * numpy.sin(met_phi)

    # Coefficients for quadratic equation for neutrino pz
    A = MW**2 / 2 + lepton_px * met_px + lepton_py * met_py
    discriminant = A**2 - (lepton_E**2 - lepton_pz**2) * (met_px**2 + met_py**2)

    if discriminant < 0:
        # If discriminant is negative, take real part (complex root scenario)
        pz_nu1 = A * lepton_pz / lepton_E**2
        pz_nu2 = pz_nu1  # Return identical solutions (use real part)
    else:
        # Two possible solutions for pz of the neutrino
        sqrt_disc = numpy.sqrt(discriminant)
        pz_nu1 = (A * lepton_pz + lepton_E * sqrt_disc) / (lepton_E**2 - lepton_pz**2)
        pz_nu2 = (A * lepton_pz - lepton_E * sqrt_disc) / (lepton_E**2 - lepton_pz**2)

    return pz_nu1, pz_nu2

def calculate_four_momentum(pt, eta, phi, mass):
    # Calculate px, py, pz
    px = pt * numpy.cos(phi)
    py = pt * numpy.sin(phi)
    pz = pt * numpy.sinh(eta)

    # Calculate energy
    E = numpy.sqrt(px**2 + py**2 + pz**2 + mass**2)

    return px, py, pz, E



class Top_chi2_reconstructionProcessor(HggBaseProcessor):
    def __init__(
        self,
        metaconditions: Dict[str, Any],
        systematics: Dict[str, List[Any]] = None,
        corrections: Dict[str, List[Any]] = None,
        apply_trigger: bool = False,
        output_location: Optional[str] = None,
        taggers: Optional[List[Any]] = None,
        trigger_group=".*DoubleEG.*",
        analysis="mainAnalysis",
        skipCQR: bool = False,
        skipJetVetoMap: bool = False,
        year: Dict[str, List[str]] = None,
        fiducialCuts: str = "classical",
        doDeco: bool = False,
        Smear_sigma_m: bool = False,
        doFlow_corrections: bool = False,
        output_format: str = "parquet"
    ) -> None:
        super().__init__(
            metaconditions,
            systematics=systematics,
            corrections=corrections,
            apply_trigger=apply_trigger,
            output_location=output_location,
            taggers=taggers,
            trigger_group=trigger_group,
            analysis=analysis,
            skipCQR=skipCQR,
            skipJetVetoMap=skipJetVetoMap,
            year=year,
            fiducialCuts=fiducialCuts,
            doDeco=doDeco,
            Smear_sigma_m=Smear_sigma_m,
            doFlow_corrections=doFlow_corrections,
            output_format=output_format
        )


    def choose_nth_object_variable(self, variable, n, fill_value):
        """
        this helper function is used to create flat collection from a jagged collection,
        parameters:
        * variable: (awkward array) selected variable from the object
        * n: (int) nth object to be selected
        * fill_value: (float) value with which to fill the padded none.
        """
        variable = variable[
            awkward.local_index(variable) == n
        ]
        variable = awkward.pad_none(
            variable, 1
        )
        variable = awkward.flatten(
            awkward.fill_none(variable, fill_value)
        )
        return variable

    def process(self, events: awkward.Array) -> Dict[Any, Any]:
        dataset_name = events.metadata["dataset"]

        # data or monte carlo?
        self.data_kind = "mc" if hasattr(events, "GenPart") else "data"

        # here we start recording possible coffea accumulators
        # most likely histograms, could be counters, arrays, ...
        histos_etc = {}
        histos_etc[dataset_name] = {}
        if self.data_kind == "mc":
            histos_etc[dataset_name]["nTot"] = int(
                awkward.num(events.genWeight, axis=0)
            )
            histos_etc[dataset_name]["nPos"] = int(awkward.sum(events.genWeight > 0))
            histos_etc[dataset_name]["nNeg"] = int(awkward.sum(events.genWeight < 0))
            histos_etc[dataset_name]["nEff"] = int(
                histos_etc[dataset_name]["nPos"] - histos_etc[dataset_name]["nNeg"]
            )
            histos_etc[dataset_name]["genWeightSum"] = float(
                awkward.sum(events.genWeight)
            )
        else:
            histos_etc[dataset_name]["nTot"] = int(len(events))
            histos_etc[dataset_name]["nPos"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["nNeg"] = int(0)
            histos_etc[dataset_name]["nEff"] = int(histos_etc[dataset_name]["nTot"])
            histos_etc[dataset_name]["genWeightSum"] = float(len(events))

        # lumi mask
        if self.data_kind == "data":
            try:
                lumimask = select_lumis(self.year[dataset_name][0], events, logger)
                events = events[lumimask]
            except:
                logger.info(
                    f"[ lumimask ] Skip now! Unable to find year info of {dataset_name}"
                )
        # apply jetvetomap: only retain events that without any jets in the EE leakage region
        if not self.skipJetVetoMap:
            events = jetvetomap(
                events, logger, dataset_name, year=self.year[dataset_name][0]
            )
        # metadata array to append to higgsdna output
        metadata = {}

        if self.data_kind == "mc":
            # Add sum of gen weights before selection for normalisation in postprocessing
            metadata["sum_genw_presel"] = str(awkward.sum(events.genWeight))
        else:
            metadata["sum_genw_presel"] = "Data"

        # apply filters and triggers
        events = self.apply_filters_and_triggers(events)

        # remove events affected by EcalBadCalibCrystal
        if self.data_kind == "data":
            events = remove_EcalBadCalibCrystal_events(events)

        # we need ScEta for corrections and systematics, it is present in NanoAODv13+ and can be calculated using PV for older versions
        events.Photon = add_photon_SC_eta(events.Photon, events.PV)

        # add veto EE leak branch for photons, could also be used for electrons
        if (
            self.year[dataset_name][0] == "2022EE"
            or self.year[dataset_name][0] == "2022postEE"
        ):
            events.Photon = veto_EEleak_flag(self, events.Photon)

        # read which systematics and corrections to process
        try:
            correction_names = self.corrections[dataset_name]
        except KeyError:
            correction_names = []
        try:
            systematic_names = self.systematics[dataset_name]
        except KeyError:
            systematic_names = []

        for correction_name in correction_names:
            if correction_name in available_object_corrections.keys():
                logger.info(
                    f"Applying correction {correction_name} to dataset {dataset_name}"
                )
                varying_function = available_object_corrections[correction_name]
                events = varying_function(
                    events=events, year=self.year[dataset_name][0]
                )
            elif correction_name in available_weight_corrections:
                # event weight corrections will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(f"Could not process correction {correction_name}.")
                continue

        original_photons = events.Photon
        # NOTE: jet jerc systematics are added in the correction functions and handled later
        original_jets = events.Jet

        # systematic object variations
        for systematic_name in systematic_names:
            if systematic_name in available_object_systematics.keys():
                systematic_dct = available_object_systematics[systematic_name]
                if systematic_dct["object"] == "Photon":
                    logger.info(
                        f"Adding systematic {systematic_name} to photons collection of dataset {dataset_name}"
                    )
                    original_photons.add_systematic(
                        # passing the arguments here explicitly since I want to pass the events to the varying function. If there is a more elegant / flexible way, just change it!
                        name=systematic_name,
                        kind=systematic_dct["args"]["kind"],
                        what=systematic_dct["args"]["what"],
                        varying_function=functools.partial(
                            systematic_dct["args"]["varying_function"],
                            events=events,
                            year=self.year[dataset_name][0],
                        )
                        # name=systematic_name, **systematic_dct["args"]
                    )
                # to be implemented for other objects here
            elif systematic_name in available_weight_systematics:
                # event weight systematics will be applied after photon preselection / application of further taggers
                continue
            else:
                # may want to throw an error instead, needs to be discussed
                warnings.warn(
                    f"Could not process systematic variation {systematic_name}."
                )
                continue

        # Applying systematic variations
        photons_dct = {}
        photons_dct["nominal"] = original_photons
        logger.debug(original_photons.systematics.fields)
        for systematic in original_photons.systematics.fields:
            for variation in original_photons.systematics[systematic].fields:
                # deepcopy to allow for independent calculations on photon variables with CQR
                photons_dct[f"{systematic}_{variation}"] = deepcopy(
                    original_photons.systematics[systematic][variation]
                )

        # NOTE: jet jerc systematics are added in the corrections, now extract those variations and create the dictionary
        jerc_syst_list, jets_dct = get_obj_syst_dict(original_jets, ["pt", "mass"])
        # object systematics dictionary
        logger.debug(f"[ jerc systematics ] {jerc_syst_list}")

        # Build the flattened array of all possible variations
        variations_combined = []
        variations_combined.append(original_photons.systematics.fields)
        # NOTE: jet jerc systematics are not added with add_systematics
        variations_combined.append(jerc_syst_list)
        # Flatten
        variations_flattened = sum(variations_combined, [])  # Begin with empty list and keep concatenating
        # Attach _down and _up
        variations = [item + suffix for item in variations_flattened for suffix in ['_down', '_up']]
        # Add nominal to the list
        variations.append('nominal')
        logger.debug(f"[systematics variations] {variations}")

        for variation in variations:
            photons, jets = photons_dct["nominal"], events.Jet
            if variation == "nominal":
                pass  # Do nothing since we already get the unvaried, but nominally corrected objets above
            elif variation in [*photons_dct]:  # [*dict] gets the keys of the dict since Python >= 3.5
                photons = photons_dct[variation]
            elif variation in [*jets_dct]:
                jets = jets_dct[variation]
            do_variation = variation  # We can also simplify this a bit but for now it works

            if self.chained_quantile is not None:
                photons = self.chained_quantile.apply(photons, events)
            # recompute photonid_mva on the fly
            if self.photonid_mva_EB and self.photonid_mva_EE:
                photons = self.add_photonid_mva(photons, events)

            # photon preselection
            photons = photon_preselection(self, photons, events, year=self.year[dataset_name][0])

            # sort photons in each event descending in pt
            # make descending-pt combinations of photons
            photons = photons[awkward.argsort(photons.pt, ascending=False)]
            photons["charge"] = awkward.zeros_like(
                photons.pt
            )  # added this because charge is not a property of photons in nanoAOD v11. We just assume every photon has charge zero...
            diphotons = awkward.combinations(
                photons, 2, fields=["pho_lead", "pho_sublead"]
            )

            # the remaining cut is to select the leading photons
            # the previous sort assures the order
            diphotons = diphotons[
                diphotons["pho_lead"].pt > self.min_pt_lead_photon
            ]

            # now turn the diphotons into candidates with four momenta and such
            diphoton_4mom = diphotons["pho_lead"] + diphotons["pho_sublead"]
            diphotons["pt"] = diphoton_4mom.pt
            diphotons["eta"] = diphoton_4mom.eta
            diphotons["phi"] = diphoton_4mom.phi
            diphotons["mass"] = diphoton_4mom.mass
            diphotons["charge"] = diphoton_4mom.charge

            diphoton_pz = diphoton_4mom.z
            diphoton_e = diphoton_4mom.energy

            diphotons["rapidity"] = 0.5 * numpy.log((diphoton_e + diphoton_pz) / (diphoton_e - diphoton_pz))

            diphotons = awkward.with_name(diphotons, "PtEtaPhiMCandidate")

            # sort diphotons by pT
            diphotons = diphotons[
                awkward.argsort(diphotons.pt, ascending=False)
            ]

            # Determine if event passes fiducial Hgg cuts at detector-level
            if self.fiducialCuts == 'classical':
                fid_det_passed = (diphotons.pho_lead.pt / diphotons.mass > 1 / 3) & (diphotons.pho_sublead.pt / diphotons.mass > 1 / 4) & (diphotons.pho_lead.pfRelIso03_all_quadratic * diphotons.pho_lead.pt < 10) & ((diphotons.pho_sublead.pfRelIso03_all_quadratic * diphotons.pho_sublead.pt) < 10) & (numpy.abs(diphotons.pho_lead.eta) < 2.5) & (numpy.abs(diphotons.pho_sublead.eta) < 2.5)
            elif self.fiducialCuts == 'geometric':
                fid_det_passed = (numpy.sqrt(diphotons.pho_lead.pt * diphotons.pho_sublead.pt) / diphotons.mass > 1 / 3) & (diphotons.pho_sublead.pt / diphotons.mass > 1 / 4) & (diphotons.pho_lead.pfRelIso03_all_quadratic * diphotons.pho_lead.pt < 10) & (diphotons.pho_sublead.pfRelIso03_all_quadratic * diphotons.pho_sublead.pt < 10) & (numpy.abs(diphotons.pho_lead.eta) < 2.5) & (numpy.abs(diphotons.pho_sublead.eta) < 2.5)
            elif self.fiducialCuts == 'none':
                fid_det_passed = diphotons.pho_lead.pt > -10  # This is a very dummy way but I do not know how to make a true array of outer shape of diphotons
            else:
                warnings.warn("You chose %s the fiducialCuts mode, but this is currently not supported. You should check your settings. For this run, no fiducial selection at detector level is applied." % self.fiducialCuts)
                fid_det_passed = diphotons.pho_lead.pt > -10

            diphotons = diphotons[fid_det_passed]

            if self.data_kind == "mc":
                # Add the fiducial flags for particle level
                diphotons['fiducialClassicalFlag'] = get_fiducial_flag(events, flavour='Classical')
                diphotons['fiducialGeometricFlag'] = get_fiducial_flag(events, flavour='Geometric')

                diphotons['PTH'], diphotons['YH'] = get_higgs_gen_attributes(events)

                genJets = get_genJets(self, events, pt_cut=30., eta_cut=2.5)
                diphotons['NJ'] = awkward.num(genJets)
                diphotons['PTJ0'] = choose_jet(genJets.pt, 0, -999.0)  # Choose zero (leading) jet and pad with -999 if none

            # baseline modifications to diphotons
            if self.diphoton_mva is not None:
                diphotons = self.add_diphoton_mva(diphotons, events)

            # workflow specific processing
            events, process_extra = self.process_extra(events)
            histos_etc.update(process_extra)

            # jet_variables
            jets = awkward.zip(
                {
                    "pt": jets.pt,
                    "eta": jets.eta,
                    "phi": jets.phi,
                    "mass": jets.mass,
                    "charge": awkward.zeros_like(
                        jets.pt
                    ),  # added this because jet charge is not a property of photons in nanoAOD v11. We just need the charge to build jet collection.
                    "hFlav": jets.hadronFlavour
                    if self.data_kind == "mc"
                    else awkward.zeros_like(jets.pt),
                    "btagDeepFlav_B": jets.btagDeepFlavB,
                    "btagDeepFlav_CvB": jets.btagDeepFlavCvB,
                    "btagDeepFlav_CvL": jets.btagDeepFlavCvL,
                    "btagDeepFlav_QG": jets.btagDeepFlavQG,
                    "jetId": jets.jetId,
                }
            )
            jets = awkward.with_name(jets, "PtEtaPhiMCandidate")

            electrons = awkward.zip(
                {
                    "pt": events.Electron.pt,
                    "eta": events.Electron.eta,
                    "phi": events.Electron.phi,
                    "mass": events.Electron.mass,
                    "charge": events.Electron.charge,
                    "cutBased": events.Electron.cutBased,
                    "mvaIso_WP90": events.Electron.mvaIso_WP90,
                    "mvaIso_WP80": events.Electron.mvaIso_WP80,
                }
            )
            electrons = awkward.with_name(electrons, "PtEtaPhiMCandidate")

            # Special cut for base workflow to replicate iso cut for electrons also for muons
            events['Muon'] = events.Muon[events.Muon.pfRelIso03_all < 0.2]

            muons = awkward.zip(
                {
                    "pt": events.Muon.pt,
                    "eta": events.Muon.eta,
                    "phi": events.Muon.phi,
                    "mass": events.Muon.mass,
                    "charge": events.Muon.charge,
                    "tightId": events.Muon.tightId,
                    "mediumId": events.Muon.mediumId,
                    "looseId": events.Muon.looseId,
                    "isGlobal": events.Muon.isGlobal,
                }
            )
            muons = awkward.with_name(muons, "PtEtaPhiMCandidate")

            # lepton cleaning
            sel_electrons = electrons[
                select_electrons(self, electrons, diphotons)
            ]
            sel_muons = muons[select_muons(self, muons, diphotons)]

            # jet selection and pt ordering
            jets = jets[
                select_jets(self, jets, diphotons, sel_muons, sel_electrons)
            ]
            jets = jets[awkward.argsort(jets.pt, ascending=False)]

            events["sel_jets"] = jets
            n_jets = awkward.num(jets)
            n_muons = awkward.num(sel_muons)
            n_electrons = awkward.num(sel_electrons)
            event_mask = (((n_muons+n_electrons) == 1) & (n_jets ==4))
            w_mask = (event_mask) #& (numpy.abs(w_delta_phi) < 0.75)
            Njets2p5 = awkward.num(jets[(jets.pt > 30) & (numpy.abs(jets.eta) < 2.5)])

            first_jet_pt = choose_jet(jets.pt, 0, -999.0)
            first_jet_eta = choose_jet(jets.eta, 0, -999.0)
            first_jet_phi = choose_jet(jets.phi, 0, -999.0)
            first_jet_mass = choose_jet(jets.mass, 0, -999.0)
            first_jet_charge = choose_jet(jets.charge, 0, -999.0)

            second_jet_pt = choose_jet(jets.pt, 1, -999.0)
            second_jet_eta = choose_jet(jets.eta, 1, -999.0)
            second_jet_phi = choose_jet(jets.phi, 1, -999.0)
            second_jet_mass = choose_jet(jets.mass, 1, -999.0)
            second_jet_charge = choose_jet(jets.charge, 1, -999.0)

            diphotons["first_jet_pt"] = first_jet_pt
            diphotons["first_jet_eta"] = first_jet_eta
            diphotons["first_jet_phi"] = first_jet_phi
            diphotons["first_jet_mass"] = first_jet_mass
            diphotons["first_jet_charge"] = first_jet_charge

            diphotons["second_jet_pt"] = second_jet_pt
            diphotons["second_jet_eta"] = second_jet_eta
            diphotons["second_jet_phi"] = second_jet_phi
            diphotons["second_jet_mass"] = second_jet_mass
            diphotons["second_jet_charge"] = second_jet_charge

            diphotons["n_jets"] = n_jets
            diphotons["Njets2p5"] = Njets2p5

            #### start of your code ###
            

            btag_sorted_indices = awkward.argsort(jets.btagDeepFlav_B, ascending=False)
            # #btag_sorted_indices = ak.argsort(jets.btagDeepFlav_B, ascending=False)
            jets = jets[btag_sorted_indices] 
            #jets = numpy.where(w_mask, jets, default_value)
            bjets = jets[:,:1]
            non_bjets = jets[:,2:]
            nonbjet_sorted_indices = awkward.argsort(non_bjets.pt, ascending=False)
            non_bjets = non_bjets[nonbjet_sorted_indices]

            # Calculate the possible jet combinations (3 jets: 2 non-bjets and 1 bjet)
            top_candidate = awkward.combinations(jets, 3, fields=("first_non_bjet", "second_non_bjet", "bjet"))

            # Calculate the four-momentum sum for top and W candidates
            top_candidate_4mom = top_candidate["first_non_bjet"] + top_candidate["second_non_bjet"] + top_candidate["bjet"]
            w_candidate_4mom = top_candidate["first_non_bjet"] + top_candidate["second_non_bjet"]

            # Fill in kinematic variables for top and W candidates
            #top_candidate_4mom = numpy.where(w_mask, top_candidate_4mom, -999)
            #w_candidate_4mom = numpy.where(w_mask, w_candidate_4mom, -999)
            # top_pt = numpy.where(w_mask, top_candidate_4mom.pt, -999)
            # top_eta = numpy.where(w_mask, top_candidate_4mom.eta, -999)
            # top_phi = numpy.where(w_mask, top_candidate_4mom.phi, -999)
            # top_mass = numpy.where(w_mask, top_candidate_4mom.mass, -999)
            # w_pt = numpy.where(w_mask, w_candidate_4mom.pt, -999)
            # w_eta = numpy.where(w_mask, w_candidate_4mom.eta, -999)
            # w_phi = numpy.where(w_mask, w_candidate_4mom.phi, -999)
            # w_mass = numpy.where(w_mask, w_candidate_4mom.mass, -999)
            top_candidate["pt"] = top_candidate_4mom.pt
            top_candidate["eta"] = top_candidate_4mom.eta
            top_candidate["phi"] = top_candidate_4mom.phi
            top_candidate["mass"] = top_candidate_4mom.mass

            # Add W boson information
            top_candidate["w_pt"] = w_candidate_4mom.pt
            top_candidate["w_eta"] = w_candidate_4mom.eta
            top_candidate["w_phi"] = w_candidate_4mom.phi
            top_candidate["w_mass"] = w_candidate_4mom.mass

            # Define the theoretical masses for the top quark and W boson
            M_top = 172.5  # GeV
            M_W = 80.4     # GeV

            # Calculate chi-squared terms for each combination
            top_chi2 = top_candidate_4mom.mass - M_top
            w_chi2 = w_candidate_4mom.mass - M_W
            top_w_chi2 = (top_candidate_4mom.mass - M_top) * (w_candidate_4mom.mass - M_W)

            # Calculate the covariance matrix elements dynamically
            v_11 = top_chi2**2
            v_12 = top_w_chi2
            v_21 = v_12  # Symmetric covariance matrix
            v_22 = w_chi2**2

            # Combine covariance matrix elements and delta vectors into structured arrays
            v_matrices = awkward.zip({
                "v11": v_11,
                "v12": v_12,
                "v21": v_21,
                "v22": v_22
            }, depth_limit=2)

            delta_vectors = awkward.zip({
                "top": top_chi2,
                "w": w_chi2
            }, depth_limit=2)

            # Define a function to compute chi-squared for each combination, careful with singular matrices
            def compute_chi2(delta, v):
                try:
                    # Convert awkward array elements to numpy arrays
                    v_matrix = numpy.array([[v["v11"], v["v12"]], [v["v21"], v["v22"]]], dtype=numpy.float64)
                    
                    # Construct the delta vector as a numpy array
                    delta_vector = numpy.array([delta["top"], delta["w"]], dtype=numpy.float64)
                    
                    # Compute the chi-squared value
                    v_inv = numpy.linalg.inv(v_matrix)  # Inverse of the covariance matrix
                    chi2 = numpy.dot(delta_vector.T, numpy.dot(v_inv, delta_vector))
                    
                    return chi2
                except numpy.linalg.LinAlgError:
                    return -999.0  # Handle singular matrix by returning a default value
            # Apply the chi-squared calculation dynamically across all events and jet combinations
            chi2_values = awkward.ArrayBuilder()

            for event_delta, event_v in zip(delta_vectors, v_matrices):
                event_chi2s = []
                for delta, v in zip(event_delta, event_v):
                    chi2 = compute_chi2(delta, v)
                    event_chi2s.append(chi2)
                chi2_values.append(event_chi2s)

            chi2_values = chi2_values.snapshot()

            # Store chi-squared values in the top candidate array
            top_candidate["top_w_chi2"] = chi2_values
            print(top_candidate.top_w_chi2)

            # Sort top candidates by their chi-squared values (ascending) for each event
            top_candidate = top_candidate[awkward.argsort(top_candidate.top_w_chi2, ascending=True, axis=1)]

            # Initialize lists to store chi-squared values
            # top_chi2_list = []

            # # Loop over each event in awkward array to compute chi-squared
            # for i in range(len(top_chi2)):
            #     top_chi2_val = top_chi2[i]
            #     print(top_chi2_val)
            #     w_chi2_val = w_chi2[i]
            #     top_w_chi2_val = top_chi2[i] * w_chi2[i]

            #     # Only process valid entries
            #     if awkward.all(~awkward.is_none(top_chi2_val)) and awkward.all(~awkward.is_none(w_chi2_val)) and awkward.all(~awkward.is_none(top_w_chi2_val)):
        # Build the covariance matrix for this combination
                    # Build the covariance matrix for this combination
                        
            #         for i in range(len(top_chi2_val)):  # Loop over each event
            #             chi2_small = []
            #             top_val = top_chi2_val[i]
            #             w_val = w_chi2_val[i]
            #             top_w_val = top_w_chi2_val[i]

            #             # Construct the v matrix for each event
            #             v = numpy.array([[top_val**2, top_w_val], [top_w_val, w_val**2]])

            #             # Construct the delta vector for each event
            #             delta = numpy.array([top_val, w_val])

            #             # Calculate the chi-squared for this event
            #             try:
            #                 chi2 = numpy.dot(delta.T, numpy.linalg.inv(v).dot(delta))  # Compute the chi-squared
            #             except numpy.linalg.LinAlgError:
            #                 print(f"Singular matrix at event {i}, skipping")
            #                 chi2 = -999.0  # Assign a default value for singular matrices
            #                 continue  # Handle the case where v is not invertible (e.g., singular matrix)
                            
            #                 # Append the chi-squared value to the list
            #             chi2_small.append(chi2)
            #         top_chi2_list.append(chi2_small)

            #     else:
            #         # Handle missing values (e.g., append NaN or None)
            #         top_chi2_list.append(None)

            # # Convert the chi-squared list back to an awkward array
            # top_chi2_ak = awkward.Array(top_chi2_list)

            # # Store the chi-squared values in the top candidate array
            # top_candidate["top_w_chi2"] = top_chi2_ak

            # # Sort top candidates based on their chi-squared values in ascending order
            # top_candidate = top_candidate[awkward.argsort(top_candidate.top_w_chi2, ascending=True)]

            # Extract the top candidate with the lowest chi-squared value
            top_candidate_pt = self.choose_nth_object_variable(top_candidate.pt, 0, -999.0)
            top_candidate_eta = self.choose_nth_object_variable(top_candidate.eta, 0, -999.0)
            top_candidate_phi = self.choose_nth_object_variable(top_candidate.phi, 0, -999.0)
            top_candidate_mass = self.choose_nth_object_variable(top_candidate.mass, 0, -999.0)

            # W candidate variables
            w_candidate_pt = self.choose_nth_object_variable(top_candidate.w_pt, 0, -999.0)
            w_candidate_eta = self.choose_nth_object_variable(top_candidate.w_eta, 0, -999.0)
            w_candidate_phi = self.choose_nth_object_variable(top_candidate.w_phi, 0, -999.0)
            w_candidate_mass = self.choose_nth_object_variable(top_candidate.w_mass, 0, -999.0)

            # Apply mask to filter candidates based on some criteria (e.g., w_mask)
            default_value = -999.0
            w_pt_masked = numpy.where(w_mask, w_candidate_pt, default_value)
            w_eta_masked = numpy.where(w_mask, w_candidate_eta, default_value)
            w_phi_masked = numpy.where(w_mask, w_candidate_phi, default_value)
            w_mass_masked = numpy.where(w_mask, w_candidate_mass, default_value)
            top_candidate_pt_masked = numpy.where(w_mask, top_candidate_pt, default_value)
            top_candidate_eta_masked = numpy.where(w_mask, top_candidate_eta, default_value)
            top_candidate_phi_masked = numpy.where(w_mask, top_candidate_phi, default_value)
            top_candidate_mass_masked = numpy.where(w_mask, top_candidate_mass, default_value)




            # Top reconstruction chi2
            
            # Calculate neutrino pz
            # met_pt = self.choose_nth_object_variable( events.MET.pt, 0, -999.0)
            # met_phi = self.choose_nth_object_variable(events.MET.phi,0, -999.0)
            # met_pt_mask = numpy.where(w_mask, met_pt, default_value)
            # met_phi_mask = numpy.where(w_mask, met_phi, default_value)
            # lepton_pt = first_lepton_pt_masked
            # lepton_eta = first_lepton_eta_masked
            # lepton_phi = first_lepton_phi_masked
            # lepton_mass = first_lepton_mass_masked
            # pz_nu1, pz_nu2 = reconstruct_neutrino_pz(lepton_pt, lepton_eta, lepton_phi, met_pt, met_phi)

            # # Calculate neutrino four-momenta
            # met_px = met_pt * numpy.cos(met_phi_mask)
            # met_py = met_pt * numpy.sin(met_phi_mask)
            # met_pz = pz_nu1  # First solution for pz
            # met_E = numpy.sqrt(met_px**2 + met_py**2 + met_pz**2)

            # #Calculate 4 momentum lepton use calculate_four_momentum
            # lepton_px, lepton_py, lepton_pz, lepton_E = calculate_four_momentum(lepton_pt, lepton_eta, lepton_phi, lepton_mass) 

            # quajets = awkward.combinations(
            #     jets, 4, fields=("first_non_bjet", "second_non_bjet", "first_bjet", "second_bjet")
            # )
            # quajets_4mom = quajets["first_non_bjet"] + quajets["second_non_bjet"] + quajets["first_bjet"] + quajets["second_bjet"]
            # quajets["pt"] = quajets_4mom.pt
            # quajets["eta"] = quajets_4mom.eta
            # quajets["phi"] = quajets_4mom.phi
            # quajets["mass"] = quajets_4mom.mass

            # if len(w_mass_masked) == 0:
            #     # Assign default values if no Ws satisfy the condition
            #     top_mass = -999.0
            # else:
            #     # Apply the same mask to the b-jet variables (assuming one-to-one correspondence)
            #     filtered_bjet_pt = numpy.where(event_mask, bjet_pt, default_value)
            #     filtered_bjet_eta = numpy.where(event_mask, bjet_eta, default_value)
            #     filtered_bjet_phi = numpy.where(event_mask, bjet_phi, default_value)
            #     filtered_bjet_mass = numpy.where(event_mask, bjet_mass, default_value)
            #     filtered_bjet_charge = numpy.where(event_mask, bjet_charge, default_value)

            #     # Calculate the four-momenta for the filtered W and b-jet candidates
            #     w_px, w_py, w_pz, w_E = calculate_four_momentum(
            #         w_pt_masked, w_eta_masked, w_phi_masked, w_mass_masked
            #     )
            #     bjet_px, bjet_py, bjet_pz, bjet_E = calculate_four_momentum(
            #         filtered_bjet_pt, filtered_bjet_eta, filtered_bjet_phi, filtered_bjet_mass
            #     )

            #     # Sum the four-momenta to get the top quark candidate
            #     top_px = w_px + bjet_px
            #     top_py = w_py + bjet_py
            #     top_pz = w_pz + bjet_pz
            #     top_E = w_E + bjet_E

            #     # Calculate the top mass
            #     top_mass = numpy.sqrt(top_E**2 - (top_px**2 + top_py**2 + top_pz**2))

            #w = w[w_mask]
            #diphotons["w_mass"] = w_mass_masked
            #diphotons["top_mass"] = top_mass
            diphotons["top_chi2"] = top_candidate_mass_masked
            diphotons["w_chi2"] = w_mass_masked
            #diphotons["dijets_delta_phi"] = w_delta_phi_masked
            # diphotons["top_charge"] = top_charge

            # run taggers on the events list with added diphotons
            # the shape here is ensured to be broadcastable
            for tagger in self.taggers:
                (
                    diphotons["_".join([tagger.name, str(tagger.priority)])],
                    tagger_extra,
                ) = tagger(
                    events, diphotons
                )  # creates new column in diphotons - tagger priority, or 0, also return list of histrograms here?
                histos_etc.update(tagger_extra)

            # if there are taggers to run, arbitrate by them first
            # Deal with order of tagger priorities
            # Turn from diphoton jagged array to whether or not an event was selected
            if len(self.taggers):
                counts = awkward.num(diphotons.pt, axis=1)
                flat_tags = numpy.stack(
                    (
                        awkward.flatten(
                            diphotons[
                                "_".join([tagger.name, str(tagger.priority)])
                            ]
                        )
                        for tagger in self.taggers
                    ),
                    axis=1,
                )
                tags = awkward.from_regular(
                    awkward.unflatten(flat_tags, counts), axis=2
                )
                winner = awkward.min(tags[tags != 0], axis=2)
                diphotons["best_tag"] = winner

                # lowest priority is most important (ascending sort)
                # leave in order of diphoton pT in case of ties (stable sort)
                sorted = awkward.argsort(diphotons.best_tag, stable=True)
                diphotons = diphotons[sorted]

            diphotons = awkward.firsts(diphotons)
            # set diphotons as part of the event record
            events[f"diphotons_{do_variation}"] = diphotons
            # annotate diphotons with event information
            diphotons["event"] = events.event
            diphotons["lumi"] = events.luminosityBlock
            diphotons["run"] = events.run
            # nPV just for validation of pileup reweighting
            diphotons["nPV"] = events.PV.npvs
            diphotons["fixedGridRhoAll"] = events.Rho.fixedGridRhoAll
            # annotate diphotons with dZ information (difference between z position of GenVtx and PV) as required by flashggfinalfits
            if self.data_kind == "mc":
                diphotons["genWeight"] = events.genWeight
                diphotons["dZ"] = events.GenVtx.z - events.PV.z
                # Necessary for differential xsec measurements in final fits ("truth" variables)
                diphotons["HTXS_Higgs_pt"] = events.HTXS.Higgs_pt
                diphotons["HTXS_Higgs_y"] = events.HTXS.Higgs_y
                diphotons["HTXS_njets30"] = events.HTXS.njets30  # Need to clarify if this variable is suitable, does it fulfill abs(eta_j) < 2.5? Probably not
                # Preparation for HTXS measurements later, start with stage 0 to disentangle VH into WH and ZH for final fits
                diphotons["HTXS_stage_0"] = events.HTXS.stage_0
            # Fill zeros for data because there is no GenVtx for data, obviously
            else:
                diphotons["dZ"] = awkward.zeros_like(events.PV.z)

            # drop events without a preselected diphoton candidate
            # drop events without a tag, if there are tags
            if len(self.taggers):
                selection_mask = ~(
                    awkward.is_none(diphotons)
                    | awkward.is_none(diphotons.best_tag)
                )
                diphotons = diphotons[selection_mask]
            else:
                selection_mask = ~awkward.is_none(diphotons)
                diphotons = diphotons[selection_mask]

            # return if there is no surviving events
            if len(diphotons) == 0:
                logger.debug("No surviving events in this run, return now!")
                return histos_etc
            if self.data_kind == "mc":
                # initiate Weight container here, after selection, since event selection cannot easily be applied to weight container afterwards
                event_weights = Weights(size=len(events[selection_mask]))

                # corrections to event weights:
                for correction_name in correction_names:
                    if correction_name in available_weight_corrections:
                        logger.info(
                            f"Adding correction {correction_name} to weight collection of dataset {dataset_name}"
                        )
                        varying_function = available_weight_corrections[
                            correction_name
                        ]
                        event_weights = varying_function(
                            events=events[selection_mask],
                            photons=events[f"diphotons_{do_variation}"][
                                selection_mask
                            ],
                            weights=event_weights,
                            dataset_name=dataset_name,
                            year=self.year[dataset_name][0],
                        )

                # systematic variations of event weights go to nominal output dataframe:
                if do_variation == "nominal":
                    for systematic_name in systematic_names:
                        if systematic_name in available_weight_systematics:
                            logger.info(
                                f"Adding systematic {systematic_name} to weight collection of dataset {dataset_name}"
                            )
                            if systematic_name == "LHEScale":
                                if hasattr(events, "LHEScaleWeight"):
                                    diphotons["nweight_LHEScale"] = awkward.num(
                                        events.LHEScaleWeight[selection_mask],
                                        axis=1,
                                    )
                                    diphotons[
                                        "weight_LHEScale"
                                    ] = events.LHEScaleWeight[selection_mask]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            elif systematic_name == "LHEPdf":
                                if hasattr(events, "LHEPdfWeight"):
                                    # two AlphaS weights are removed
                                    diphotons["nweight_LHEPdf"] = (
                                        awkward.num(
                                            events.LHEPdfWeight[selection_mask],
                                            axis=1,
                                        )
                                        - 2
                                    )
                                    diphotons[
                                        "weight_LHEPdf"
                                    ] = events.LHEPdfWeight[selection_mask][
                                        :, :-2
                                    ]
                                else:
                                    logger.info(
                                        f"No {systematic_name} Weights in dataset {dataset_name}"
                                    )
                            else:
                                varying_function = available_weight_systematics[
                                    systematic_name
                                ]
                                event_weights = varying_function(
                                    events=events[selection_mask],
                                    photons=events[f"diphotons_{do_variation}"][
                                        selection_mask
                                    ],
                                    weights=event_weights,
                                    dataset_name=dataset_name,
                                    year=self.year[dataset_name][0],
                                )

                diphotons["weight_central"] = event_weights.weight()
                # Store variations with respect to central weight
                if do_variation == "nominal":
                    if len(event_weights.variations):
                        logger.info(
                            "Adding systematic weight variations to nominal output file."
                        )
                    for modifier in event_weights.variations:
                        diphotons["weight_" + modifier] = event_weights.weight(
                            modifier=modifier
                        )

                # Multiply weight by genWeight for normalisation in post-processing chain
                event_weights._weight = (
                    events["genWeight"][selection_mask]
                    * diphotons["weight_central"]
                )
                diphotons["weight"] = event_weights.weight()

            # Add weight variables (=1) for data for consistent datasets
            else:
                diphotons["weight_central"] = awkward.ones_like(
                    diphotons["event"]
                )
                diphotons["weight"] = awkward.ones_like(diphotons["event"])

            if self.output_location is not None:
                if self.output_format == "root":
                    df = diphoton_list_to_pandas(self, diphotons)
                else:
                    akarr = diphoton_ak_array(self, diphotons)

                    # Remove fixedGridRhoAll from photons to avoid having event-level info per photon
                    akarr = akarr[
                        [
                            field
                            for field in akarr.fields
                            if "lead_fixedGridRhoAll" not in field
                        ]
                    ]

                fname = (
                    events.behavior[
                        "__events_factory__"
                    ]._partition_key.replace("/", "_")
                    + ".%s" % self.output_format
                )
                subdirs = []
                if "dataset" in events.metadata:
                    subdirs.append(events.metadata["dataset"])
                subdirs.append(do_variation)
                if self.output_format == "root":
                    dump_pandas(self, df, fname, self.output_location, subdirs)
                else:
                    dump_ak_array(
                        self, akarr, fname, self.output_location, metadata, subdirs,
                    )

        return histos_etc

    def process_extra(self, events: awkward.Array) -> awkward.Array:
        return events, {}

    def postprocess(self, accumulant: Dict[Any, Any]) -> Any:
        pass

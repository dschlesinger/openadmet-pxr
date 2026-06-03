"""Molecular representation registry for the OpenADMET PXR challenge.

To add a new representation: create a module in this directory, subclass
Representation, set a unique `name` class variable, and add it here.
"""

from representations.base import Representation
from representations.chemeleon import ChemeleonFingerprint, FinetunedChemeleonFingerprint
from representations.chemprop_rep import ChempropFingerprint
from representations.fingerprints import MorganFingerprint
from representations.jazzy_descriptors import JazzyDescriptors
from representations.mole import MolERepresentation
from representations.rdkit_descriptors import RDKitDescriptors
from representations.unimol import UniMolRepresentation
from representations.unimol_finetuned import FinetunedUniMolRepresentation
from representations.xtb_descriptors import XTBDescriptors

REGISTRY: dict[str, type[Representation]] = {
    MorganFingerprint.name: MorganFingerprint,
    RDKitDescriptors.name: RDKitDescriptors,
    ChemeleonFingerprint.name: ChemeleonFingerprint,
    ChempropFingerprint.name: ChempropFingerprint,
    FinetunedChemeleonFingerprint.name: FinetunedChemeleonFingerprint,
    UniMolRepresentation.name: UniMolRepresentation,
    FinetunedUniMolRepresentation.name: FinetunedUniMolRepresentation,
    MolERepresentation.name: MolERepresentation,
    JazzyDescriptors.name: JazzyDescriptors,
    XTBDescriptors.name: XTBDescriptors,
}

__all__ = ["Representation", "REGISTRY"]

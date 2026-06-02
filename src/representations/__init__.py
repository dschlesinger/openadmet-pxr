"""Molecular representation registry for the OpenADMET PXR challenge.

To add a new representation: create a module in this directory, subclass
Representation, set a unique `name` class variable, and add it here.
"""

from representations.base import Representation
from representations.chemeleon import ChemeleonFingerprint, FinetunedChemeleonFingerprint
from representations.chemprop_rep import ChempropFingerprint
from representations.fingerprints import MorganFingerprint
from representations.rdkit_descriptors import RDKitDescriptors
from representations.mole import MolERepresentation
from representations.unimol import UniMolRepresentation
from representations.unimol_finetuned import FinetunedUniMolRepresentation

REGISTRY: dict[str, type[Representation]] = {
    MorganFingerprint.name: MorganFingerprint,
    RDKitDescriptors.name: RDKitDescriptors,
    ChemeleonFingerprint.name: ChemeleonFingerprint,
    ChempropFingerprint.name: ChempropFingerprint,
    FinetunedChemeleonFingerprint.name: FinetunedChemeleonFingerprint,
    UniMolRepresentation.name: UniMolRepresentation,
    FinetunedUniMolRepresentation.name: FinetunedUniMolRepresentation,
    MolERepresentation.name: MolERepresentation,
}

__all__ = ["Representation", "REGISTRY"]

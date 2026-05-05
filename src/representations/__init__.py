"""Molecular representation registry for the OpenADMET PXR challenge.

To add a new representation: create a module in this directory, subclass
Representation, set a unique `name` class variable, and add it here.
"""

from representations.base import Representation
from representations.chemeleon import ChemeleonFingerprint
from representations.fingerprints import MorganFingerprint
from representations.rdkit_descriptors import RDKitDescriptors

REGISTRY: dict[str, type[Representation]] = {
    MorganFingerprint.name: MorganFingerprint,
    RDKitDescriptors.name: RDKitDescriptors,
    ChemeleonFingerprint.name: ChemeleonFingerprint,
}

__all__ = ["Representation", "REGISTRY"]

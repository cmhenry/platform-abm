"""Institution strategy implementations."""

from platform_abm.institutions.algorithmic import AlgorithmicInstitution
from platform_abm.institutions.base import Institution
from platform_abm.institutions.coalition import CoalitionInstitution
from platform_abm.institutions.direct import DirectInstitution

INSTITUTION_REGISTRY: dict[str, type[Institution]] = {
    "direct": DirectInstitution,
    "coalition": CoalitionInstitution,
    "algorithmic": AlgorithmicInstitution,
}

__all__ = [
    "Institution",
    "DirectInstitution",
    "CoalitionInstitution",
    "AlgorithmicInstitution",
    "INSTITUTION_REGISTRY",
]

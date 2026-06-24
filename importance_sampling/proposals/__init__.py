from .base import SamplePathLengthProposal
from .geometric import GeometricLengthProposal, TruncatedGeometricLengthProposal
from .fixed import FixedLengthProposal
from .arrival_generator_proposal import ArrivalGeneratorSamplePathProposal
from .mixture import MixtureGeometricStratifiedQMCProposal

# Spec-type string -> proposal class. ``build_proposal`` uses this to turn the
# JSON-style proposal specs stored in experiment configs into instances. To add
# a proposal: create its module, import the class above, add one line here.
_PROPOSAL_TYPES = {
    'geometric': GeometricLengthProposal,
    'truncated_geometric': TruncatedGeometricLengthProposal,
    'fixed': FixedLengthProposal,
    'arrival_generator': ArrivalGeneratorSamplePathProposal,
    'mixture_geometric': MixtureGeometricStratifiedQMCProposal,
}


def build_proposal(spec):
    """Build a proposal from a spec.

    ``spec`` is one of:
      * ``None`` -> returns ``None`` (no importance sampling),
      * a ``SamplePathLengthProposal`` instance -> returned unchanged,
      * a ``dict`` with a ``'type'`` key plus constructor keyword arguments.
    """
    if spec is None:
        return None
    if isinstance(spec, SamplePathLengthProposal):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f'Unsupported proposal spec: {spec!r}')
    spec = dict(spec)
    proposal_type = spec.pop('type', None)
    if proposal_type is None:
        raise ValueError("Proposal spec dict must include a 'type' key.")
    if proposal_type not in _PROPOSAL_TYPES:
        raise ValueError(
            f'Unknown proposal type {proposal_type!r}. '
            f'Available: {sorted(_PROPOSAL_TYPES)}'
        )
    return _PROPOSAL_TYPES[proposal_type](**spec)


__all__ = [
    'SamplePathLengthProposal',
    'GeometricLengthProposal',
    'TruncatedGeometricLengthProposal',
    'FixedLengthProposal',
    'ArrivalGeneratorSamplePathProposal',
    'MixtureGeometricStratifiedQMCProposal',
    'build_proposal',
]

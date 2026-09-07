"""Normalization and construction of penalty generating functions."""

from generating_function import AbsorptionLinearPenaltyFunction, LinearPenaltyFunction

GENERATING_FUNCTION_CLASSES = {
    cls.spec_name: cls for cls in (LinearPenaltyFunction, AbsorptionLinearPenaltyFunction)
}


def normalize_generating_function_spec(spec):
    if spec is None:
        return {'name': 'linear_penalty'}
    if isinstance(spec, str):
        return {'name': spec}
    if isinstance(spec, dict):
        spec = dict(spec)
        if 'name' not in spec and 'type' in spec:
            spec['name'] = spec.pop('type')
        spec.setdefault('name', 'linear_penalty')
        return spec
    raise ValueError(f"Unsupported generating function spec: {spec}")


def build_generating_function(env, spec, coefficients=None):
    spec = normalize_generating_function_spec(spec)
    name = spec.get('name')
    if coefficients is None:
        coefficients = spec.get('coefficients')
    if name not in GENERATING_FUNCTION_CLASSES:
        raise ValueError(f"Unsupported generating function name: {name}; use one of {sorted(GENERATING_FUNCTION_CLASSES)}")
    generating_function = GENERATING_FUNCTION_CLASSES[name](env=env, coefficients=coefficients)
    if coefficients is None:
        # No coefficients supplied anywhere -> treat all coefficients as zeros.
        generating_function.set_coefficients([0.0] * generating_function.number_of_coefficients)
    return generating_function

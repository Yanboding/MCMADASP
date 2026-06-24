"""Normalization and construction of penalty generating functions."""

from generating_function import LinearPenaltyFunction, MulticlassQuadraticPenaltyFunction


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
    if name == 'linear_penalty':
        generating_function = LinearPenaltyFunction(env=env, coefficients=coefficients)
    elif name in {'multiclass_quadratic_penalty', 'quadratic_penalty'}:
        generating_function = MulticlassQuadraticPenaltyFunction(env=env, coefficients=coefficients)
    else:
        raise ValueError(f"Unsupported generating function name: {name}")
    if coefficients is None:
        # No coefficients supplied anywhere -> treat all coefficients as zeros.
        generating_function.set_coefficients([0.0] * generating_function.number_of_coefficients)
    return generating_function

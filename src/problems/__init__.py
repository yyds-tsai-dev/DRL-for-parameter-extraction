"""Built-in problem registrations.

Importing this package registers every built-in problem exactly once.
Third-party problems call :func:`problems.registry.register` themselves.
"""

from problems import eehemt, hardness, registry

for _builder in (hardness.build_spec, eehemt.build_spec):
    _spec = _builder()
    if _spec.name not in registry.names():
        registry.register(_spec)

del _builder, _spec

"""
Tests for the model's pipeline overrides.

The hybrid model overrides one hook on one stage, which is the whole point
of the four customisation tiers: a model whose only quarrel with the base
pipeline is the list of reports it renders should not have to reimplement a
stage to express that.

What these tests mostly assert is therefore an *absence* -- that no stage is
replaced, that the sequence is unchanged, that the user's own selection
survives. An override that grew beyond this is a finding about the
framework's step granularity rather than about this model.
"""

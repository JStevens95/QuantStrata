"""
Tests for the network's architectural blocks, one module per block.

Each block is tested in isolation: given an input of a known shape, assert
the output shape, that gradients reach every parameter, and the invariances
the block is supposed to have. Testing an architecture only end to end makes
every shape bug a bisection exercise, and makes a block that quietly
contributes nothing indistinguishable from one that works.
"""

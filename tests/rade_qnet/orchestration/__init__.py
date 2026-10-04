"""
Tests for ``rade_qnet.orchestration`` -- pipelines, job sets and placement.

The property this sub-tree protects is that *placement cannot change results*.
A job set run sequentially and the same job set run across eight processes must
produce identical artifacts, and the only way to be sure is to run both and
compare. Several tests here do exactly that.
"""

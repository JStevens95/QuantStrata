"""
Tests for the execution layer.

Almost everything here is testing a property that holds *across* executors
rather than a property of one of them. That shape is deliberate: the
proposition this phase rests on is that placement cannot change results, and
a test written against one executor cannot say anything about it.

So the conformance rules -- input ordering, returned failures, repeatability
-- are parametrised over every executor, and the sequential one is treated as
the reference the others are compared against.

Tests that spawn processes are kept deliberately cheap: trivial payloads, two
workers, no model. A pool test slow enough to be annoying is a pool test
somebody eventually marks as skipped, and the gate goes with it. The one
place a real model runs under a real pool is parity level 5, which runs once.
"""

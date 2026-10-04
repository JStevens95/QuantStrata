"""
Tests for ``rade_qnet.analysis`` -- metrics, visuals and reports.

Figures are rendered through a non-interactive backend so the suite needs no
display and opens no windows. Tests assert on figure structure -- axis labels,
series count, data limits -- rather than on pixels, which is both stable across
library versions and actually informative when it fails.
"""

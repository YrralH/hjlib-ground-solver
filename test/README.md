# Data-dependent tests

This tracked placeholder keeps the mandatory `test/` tree explicit. The first
data-dependent case is `get_ground_by_smpls_on_the_ground` with real SMPL model
weights and vertices; until that fixture is available, portable coverage stays
in `test_smoke/` and migration parity stays in `hjlib-migration-tests`.

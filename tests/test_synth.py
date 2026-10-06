import pandas as pd

from ayojna.ingest.synth import PROFILES, generate


def test_generate_writes_msr_format(tmp_path):
    paths = generate(tmp_path, days=1, seed=1)
    assert len(paths) == len(PROFILES)
    df = pd.read_csv(paths[0], header=None)
    assert df.shape[1] == 7                      # MSR has 7 columns
    assert set(df[3].unique()) <= {"Read", "Write"}
    assert df[0].is_monotonic_increasing          # timestamps in order

import pytest
from src.alpha_brain import HFTAlphaSignals

def test_win_rate_simulation():
    import random
    brain = HFTAlphaSignals(obi_threshold=0.70)
    # let's set maxlen to 2
    brain.obi_history = __import__('collections').deque(maxlen=2)
    # patch the check_signals to use < 2
    original_check = brain.check_signals
    # actually let's just modify the file and run pytest

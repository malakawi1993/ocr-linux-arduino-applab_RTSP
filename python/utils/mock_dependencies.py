"""
Mock dependencies to avoid import errors on Arduino UNO Q.

The edge_impulse_linux package has transitive dependencies on 'six' and 'pyaudio'
that may not be available or needed on the UNO Q. This module patches sys.modules
to prevent ImportError when those packages are imported internally.

Call apply_mocks() BEFORE importing edge_impulse_linux.
"""

import sys
import builtins


def apply_mocks():
    """Apply mocks for six and pyaudio to avoid dependency issues."""

    # Mock six.moves.queue (Python 2/3 compat library, not needed on Python 3.10+)
    class MockSixMovesQueue:
        Queue = None

    class MockSixMoves:
        queue = MockSixMovesQueue()

    class MockSix:
        moves = MockSixMoves()

    sys.modules['six'] = MockSix()
    sys.modules['six.moves'] = MockSixMoves()
    sys.modules['six.moves.queue'] = MockSixMovesQueue()

    # Mock pyaudio (audio library, not needed for vision-only OCR)
    class MockPyAudio:
        pass

    sys.modules['pyaudio'] = MockPyAudio()
    builtins.pyaudio = MockPyAudio()

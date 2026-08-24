"""Pytest configuration and custom CLI flags for emdgrid tests."""


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--strict-pot",
        action="store_true",
        default=False,
        help="Enforce strict transport plan equivalence against POT",
    )

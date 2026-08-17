"""Diamond 3-player controller console.

Layers, strictly one-directional::

    QML  ->  app (GameController, Qt models, AI worker)  ->  game (rules)
                                   ^
                                   |
                               agents (Agent protocol)

``game`` and ``agents`` never import PySide6, so the whole engine is headless
and unit-testable.
"""

__version__ = "0.1.0"

from .api import Chembl, Papyrus

__all__ = ["Chembl", "Papyrus"]

if __name__ == "__main__":
    from .papyrus import ensure_raw, latest

    print(latest())
    ensure_raw(latest())

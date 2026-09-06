from .dataset import Adrecs, AdrecsTarget, Chembl, Drugbank, Papyrus, Sider

__all__ = ["Adrecs", "AdrecsTarget", "Chembl", "Drugbank", "Papyrus", "Sider"]

if __name__ == "__main__":
    from .papyrus import ensure_raw_files, latest

    print(latest())
    ensure_raw_files(latest())

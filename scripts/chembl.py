import logging as lg

from fairfetched.get import Chembl

if __name__ == "__main__":
    lg.basicConfig(level=lg.DEBUG)
    Chembl.from_latest()
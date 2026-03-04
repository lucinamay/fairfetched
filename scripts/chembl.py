import logging as lg

from fairfetched.get.chembl import ensure

if __name__ == "__main__":
    lg.basicConfig(level=lg.DEBUG)
    ensure()

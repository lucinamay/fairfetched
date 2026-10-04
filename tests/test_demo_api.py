import pytest

from fairfetched.get.dataset import (
    Chembl,
    Papyrus,
    PubchemBioassay,
    PubchemCompound,
    Toxcast,
)

DEMOS = [Chembl, Papyrus, Toxcast, PubchemBioassay, PubchemCompound]


@pytest.mark.parametrize("cls", DEMOS)
class TestDemoMetadata:
    def test_sources_empty_offline(self, cls):
        assert cls.demo().sources == {}

    def test_hashable(self, cls):
        assert isinstance(hash(cls.demo()), int)

    def test_equal_demos_hash_equal(self, cls):
        a, b = cls.demo(), cls.demo()
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


def test_different_datasets_not_equal():
    assert Chembl.demo() != Papyrus.demo()

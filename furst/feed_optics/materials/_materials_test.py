import optika
import furst


def test_coating_design():
    r = furst.feed_optics.materials.coating_design()
    assert isinstance(r, optika.materials.AbstractMultilayerMirror)


def test_coating_witness_measured():
    r = furst.feed_optics.materials.coating_witness_measured()
    assert isinstance(r, optika.materials.MeasuredMirror)


def test_coating_witness_fit():
    r = furst.feed_optics.materials.coating_witness_fit()
    assert isinstance(r, optika.materials.MultilayerMirror)

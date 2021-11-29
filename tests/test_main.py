from unittest import TestCase

from main import fun


class Test(TestCase):
    def test_fun(self):
        nft = fun('nft')
        self.assertEquals(nft, 'input: nft')

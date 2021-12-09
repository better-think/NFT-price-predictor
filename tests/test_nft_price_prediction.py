from os.path import join
from unittest import TestCase

from nft_price_prediction import run


class Test(TestCase):
    def test_run(self):
        run(join('resources', 'run-data', 'images'), join('resources', 'run-data', 'nft'))

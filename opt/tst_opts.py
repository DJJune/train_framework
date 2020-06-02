from .base_opts import BaseOptions


class TestOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--eval_epoch', type=int, help='which epoch to restore')

        self.isTrain = False
        return parser

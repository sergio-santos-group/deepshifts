from deepshifts.cli import train,infer,serve,hyperband
from deepshifts.cli import descriptors
from deepshifts import indexing
import argparse

desc = '''
Train an ANN model for predicting 1H chemical shifts.
Monitoring of the training status can be tracked by mlflow
(mlflow.org) and tensorboad.
'''

class FromFileArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

parser = FromFileArgumentParser(
    prog='deepshifts',
    description=desc,
    fromfile_prefix_chars='@',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

sub_parsers = parser.add_subparsers(title='subcommands',
                #description='valid subcommands',
                help='command selector')


for module in (train, infer, serve, descriptors, indexing, hyperband):
    get_arg_parser = getattr(module, 'get_arg_parser')
    subparser = get_arg_parser(sub_parsers)

args = parser.parse_args()

func = args.func
del args.func

exit(func(args))



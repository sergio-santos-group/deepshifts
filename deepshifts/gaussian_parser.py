from tqdm import tqdm
import numpy as np
import hashlib
import gzip
import json
import sys
import re

__version__ = '0.1.0'

ATNUM2SYMB = {
    '1': 'H',
    '6': 'C',
    '7': 'N',
    '8': 'O'
}


#--------------------------------------------------------------------
# BaseParser
#--------------------------------------------------------------------
class BaseParser(object):
    @classmethod
    def check(cls, line):
        if not cls.match(line):
            raise Exception(f'Invalid "{cls.__name__}" block')


#--------------------------------------------------------------------
# LinkParser
#--------------------------------------------------------------------
class Link1Parser(BaseParser):
    @staticmethod
    def match(line):
        return line.startswith(' Entering Link 1 =')


#--------------------------------------------------------------------
# TerminationParser
#--------------------------------------------------------------------
class TerminationParser(BaseParser):
    @staticmethod
    def match(line):
        return line.startswith(' Normal termination of Gaussian')


#--------------------------------------------------------------------
# TitleParser
#--------------------------------------------------------------------
class TitleParser(BaseParser):
    _match_re = re.compile(r'^ 99/9=1/99;')
    _parse_re3 = re.compile(r'^{.*}$', re.IGNORECASE)
    _parse_re2 = re.compile(r'^SMILES:\s+(?P<smiles>\S+)\s+(?P<path>\S+)\s+(?P<index>\d+)\s+nmr$', re.IGNORECASE)
    _parse_re1 = re.compile(r'^(?P<smiles>\S+)\s+nmr$', re.IGNORECASE)

    @staticmethod
    def match(line):
        '''
         99/9=1/99;
        '''
        return TitleParser._match_re.match(line)
    
    @staticmethod
    def parse(line, iterator, *args):
        '''
         99/9=1/99;
         ----------
         <title string>+
         ----------

        v1: NCCN nmr
        v2: SMILES: [H]N([H])[H] /gdb11_s01/gdb11_s01-1 1107 nmr
        v3: { "smiles": "[H]OO[H]", "path": "/gdb11_s02/gdb11_s02-6", "index": 2715 }

        '''
        TitleParser.check(line)
        line = next(iterator)
        termination = f' {len(line.strip())*"-"}'
        line = next(iterator)
        
        title = ''
        while not line.startswith(termination):
            title += line[1:-1]#.strip()
            line = next(iterator)
        
        # JSON-styled title card (version 3)
        if TitleParser._parse_re3.match(title):
            fields = json.loads(title)
        else:
            # text-styled extended title card (version 2)
            match = TitleParser._parse_re2.match(title)
            if match is None:
                # text-styled simple title card (version 1)
                match = TitleParser._parse_re1.match(title)
            if match is not None:
                fields = match.groupdict()
            else:
                fields = {'text': title}
        
        fields['index'] = int(fields.get('index', -1))
        return fields


#--------------------------------------------------------------------
# EnergyParser
#--------------------------------------------------------------------
class EnergyParser(BaseParser):
    _match_re = re.compile(r'^ SCF Done:[\s\w()]+=\s*(?P<energy>-?\d+\.\d+)\s*')
    
    @staticmethod
    def match(line):
        '''
         SCF Done:  E(RB3LYP) =  -213.766066808     A.U. after   12 cycles
        '''
        return EnergyParser._match_re.match(line)
    
    @staticmethod
    def parse(line, iterator, *args):
        '''
         SCF Done:  E(RB3LYP) =  -213.766066808     A.U. after   12 cycles
        '''
        EnergyParser.check(line)
        energy = EnergyParser._match_re.search(line).group('energy')
        return float(energy)


#--------------------------------------------------------------------
# ShiftsParser
#--------------------------------------------------------------------
class ShiftsParser(BaseParser):
    #_parse_re = re.compile(r'.*Isotropic =\s*(?P<shift>-?\d+\.\d+)\s*Anisotropy')
    _parse_re = re.compile(r'.*Isotropic =\s*(?P<iso>-?\d+\.\d+)\s*Anisotropy =\s*(?P<aniso>-?\d+\.\d+)')
    
    @staticmethod
    def match(line):
        return line.startswith(' SCF GIAO Magnetic shielding tensor (ppm):')
        
    @staticmethod
    def parse(line, iterator, expected):
        ShiftsParser.check(line)
        isotropic = []
        anisotropic = []
        while len(isotropic) < expected:
            r = None
            while r is None:
                r = ShiftsParser._parse_re.search(line)
                line = next(iterator)
            isotropic.append(float(r.group('iso')))
            anisotropic.append(float(r.group('aniso')))
        
        return np.array(isotropic), np.array(anisotropic)


#--------------------------------------------------------------------
# OrientationParser
#--------------------------------------------------------------------
class OrientationParser(BaseParser):
    _match_re = re.compile(r'^\s+Input orientation:\s+$')
    
    @staticmethod
    def match(line):
        return OrientationParser._match_re.match(line)
        
    @staticmethod
    def parse(line, iterator, *args):
        '''
                                  Input orientation:                          
         ---------------------------------------------------------------------
         Center     Atomic      Atomic             Coordinates (Angstroms)
         Number     Number       Type             X           Y           Z
         ---------------------------------------------------------------------
         <orientation string>+
         ---------------------------------------------------------------------
        '''
        OrientationParser.check(line)
            
        # skip next 5 lines (header lines)
        for _ in range(5):
            line = next(iterator)
        
        digester = hashlib.sha512()
        atoms = []
        while not line.startswith(' ------'):
            digester.update(line.encode('utf8'))
            atoms.append(line.split())
            line = next(iterator)
        
        species = [ATNUM2SYMB[at[1]] for at in atoms]
        coords = np.array([at[3:] for at in atoms], dtype=np.float)
        return species, coords, digester.digest().hex()


#--------------------------------------------------------------------
# GaussianLogParser
#--------------------------------------------------------------------
class GaussianLogParser(object):
    
    def __init__(self, max_confs=-1, stride=1):
        self.max_confs = max_confs
        self.stride = stride
    
    def _parse(self, fhandle):

        is_new_conf = False
        iterator = iter(fhandle)
        configurations = []
        skip = False
        counter = 0

        # read first line and check if it is a valid
        # Gaussian output file. If not, return an empty list
        line = next(iterator)
        if not line.startswith(' Entering Gaussian System'):
            return []
        
        while True:
            try:
                line = next(iterator)
            except StopIteration:
                break

            try:
                
                if self._match(Link1Parser, line, not is_new_conf):
                    skip = counter % self.stride > 0
                    is_new_conf = True
                    counter += 1
                    n_fields = 0
                
                elif skip:
                    continue
                
                elif self._match(TitleParser, line, is_new_conf):
                    title = TitleParser.parse(line, iterator)
                    n_fields += 1
                
                elif self._match(OrientationParser, line, is_new_conf):
                    species, coords, signature = OrientationParser.parse(line, iterator)
                    n_fields += 3

                elif self._match(EnergyParser, line, is_new_conf):
                    energy = EnergyParser.parse(line, iterator)
                    n_fields += 1
                
                elif self._match(ShiftsParser, line, is_new_conf):
                    shifts,anisotropic = ShiftsParser.parse(line, iterator, len(species))
                    n_fields += 1
                
                elif self._match(TerminationParser, line, is_new_conf):
                    if n_fields != 6:
                        tqdm.write(f'ERROR in {self._current_file}: reached a Termination flag without all required sections {n_fields}!', file=sys.stdout)
                        break

                    configuration = {
                        'signature': signature,
                        'species'  : species,
                        'coords'   : coords,
                        'energy'   : energy,
                        'shifts'   : shifts,
                        'anisotropic'   : anisotropic,
                        'title'    : title,
                    }
                    configurations.append(configuration)
                    is_new_conf = False
                    if len(configurations) == self.max_confs:
                        break
            
            except Exception as e:
                tqdm.write(str(e), file=sys.stdout)
                break
        
        if is_new_conf:
            tqdm.write(f'ERROR in {self._current_file}: Reached end of file without finding a Termination flag!', file=sys.stdout)

        return configurations


    def _match(self, parser, line, is_within_block):
        matches = parser.match(line)
        if matches and not is_within_block:
            raise Exception(f'ERROR in {self._current_file}: trying to call {parser.__name__} outside block')
        return matches
    

    def parse(self, fname):
        self._current_file = fname
        try:
            if fname.endswith('.gz'):
                fin = gzip.open(fname, 'rt')
            else:
                fin = open(fname)
        except IOError as e:
            raise e
        else:   
            output = self._parse(fin)
            fin.close()
        return output



#--------------------------------------------------------------------
# MAIN
#--------------------------------------------------------------------
def main(args):
    from deepshifts.utils import file_finder
    from deepshifts import h5handler
    # import json
    import operator
    import os
    
    
    files = file_finder(args.files, '.log', '.log.gz')
    
    # given a path to a file: 
    #      /path/to/file/groupname<delimiter>ID.log[.gz]
    # split it into:
    #      groupname<delimiter>ID.log[.gz]
    #          and
    #      /path/to/file/ 
    files = [(os.path.basename(f),f) for f in files]
    
    # take all unique group names
    groups = sorted(set(f.split(args.delimiter)[0] for f,_ in files))

    gparser = GaussianLogParser(args.max_confs, args.stride)
    Handler = h5handler.StdoutHandler if args.stdout else h5handler.H5Handler
    signatures = set()
    mode = 'a' if args.append else 'w-'

    with Handler(args.out, mode=mode) as h5handler, \
         tqdm(groups, leave=False) as gbar, \
         tqdm(files, leave=False) as fbar:

        # process multiple files by group name (each group might be
        # split into multiple parts)
        for group in groups:
            gbar.set_description(group)
            gbar.update(1)

            group_members = [f for b,f in files if b.startswith(group + args.delimiter)]
            group_members.sort()
            
            items = []
            signatures = set() 
            for filename in group_members:
                # update progress bar
                # fbar.set_description(filename)
                fbar.update(1)

                # parse files
                item = gparser.parse(filename)
                items.extend(item)
            
            # filter unique items
            unique = []
            for n,item in enumerate(items):
                signature = item['signature']
                if signature in signatures:
                    tqdm.write(f'ERROR found duplicate for item {n} in group {group}: {item["title"]}', file=sys.stdout)
                else:
                    unique.append(item)
                signatures.add(signature)
            items = unique

            if not items:
                continue
            
            items.sort(key=lambda item: item['title'].get('index',-1))
            #items.sort(key=operator.itemgetter('index'))
            
            # save parser output to file
            #try:
            h5handler.save(
                group
                , coords=np.array([d['coords'] for d in items], dtype=np.float)
                , energy=np.array([d['energy'] for d in items], dtype=np.float).reshape(-1,1)
                , shifts=np.array([d['shifts'] for d in items], dtype=np.float)
                , anisotropic=np.array([d['anisotropic'] for d in items], dtype=np.float)
                , index =np.array([d['title'].get('index',-1) for d in items], dtype=np.int)
                #, smiles=items[0].get('smiles', 'UNK')
                , smiles=items[0]['title'].get('smiles', 'UNK')
                , species=items[0]['species']
            )
            #except Exception as e:
            #    tqdm.write(str(items[0]['title']), file=sys.stderr)
            #    tqdm.write(str(items[0]['species']), file=sys.stderr)
            #    raise e
        fbar.write('All done')
    


#--------------------------------------------------------------------
#--------------------------------------------------------------------
if __name__ == "__main__":
    from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
    
    groupname = '\033[1mGROUPNAME\033[0m'
    delimiter = '\033[4mdelimiter\033[0m'
    # -------------------------------------------
    # input argument parser
    # -------------------------------------------
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--stride', type=int, default=1, metavar='N',
        help='extract only every  N-th configuration', action='store')
    parser.add_argument('--max-confs', type=int, default=-1, metavar='M',
        help='maximum number of configurations to extract from single file', action='store')
    parser.add_argument('--out', type=str, default='data.h5', action='store', metavar='FILE',
        help='output filename')
    parser.add_argument('--delimiter', type=str, default='_part',
        help=(f'delimiter for identifying {groupname} in /path/to/file/{groupname}<{delimiter}>ID.log[.gz]. '
              f'Files with the same {groupname} will be merged into a single group.'))
    parser.add_argument('--stdout', action='store_true',
        help='write to stdout')
    parser.add_argument('files', type=str, nargs='+',
        help=(f'Gaussian output files for parsing. May be plain-text or in gzip format. '
              f'If one (or more) directory is provided, all files ending in *.{{log,log.gz}} will '
              f'be recursively found and processed'))
    parser.add_argument('--append', action='store_true', help='open h5 file in append-mode')
    args = parser.parse_args()
    

    # -------------------------------------------
    # main block
    # -------------------------------------------
    main(args)

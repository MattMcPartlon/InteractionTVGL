import _pickle as cPickle
import os
import sys
import numpy as np
import PDBUtils
from SequenceUtils import LoadFASTAFile

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('python CalcGroundTruthFromSeqPDB.py structFile seqFile [resDir]')
        print(
            '	This script calculates protein properties, distance and orientation matrices from a protein structure file and a sequence file')
        print('	structFile: a protein structure file ending with .cif or .pdb')
        print(
            '	seqFile: a file for primary sequence in FASTA format. It is used to determine which residues shall be selected from the structure file')
        print('	resDir: the folder for result saving, default current work directory')
        print(
            '		the resultant file is named after targetName.native.pkl where targetName is the basename of seqFile')
        exit(1)
    pdbfile = None
    structfile = sys.argv[1]
    name = os.path.basename(structfile).split('.')[0]
    sequence = None
    if len(sys.argv) >= 3:
        seqfile = sys.argv[2]
        if os.path.isfile(seqfile):
            sequence = LoadFASTAFile(seqfile)
            name = os.path.basename(seqfile).split('.')[0]

    resDir = os.getcwd()
    if len(sys.argv) >= 4:
        resDir = sys.argv[3]
        if not os.path.isdir(resDir):
            print('ERROR: the result directory does not exist: ', resDir)
            exit(1)

    if structfile.endswith('.pdb'):
        pdbfile = structfile
        pdbfileIsTemporary = False
    else:
        print('ERROR: the input structure file shall end with .cif or .pdb')
        exit(1)

    protein = dict()
    protein['name'] = name

    protein['sequence'] = sequence
    protein['seq4matrix'] = sequence
    result, pdbseq, numMisMatches, numMatches = PDBUtils.ExtractCoordinatesNDSSPBySeq(sequence, pdbfile)
    if numMisMatches > 5:
        print('ERROR: too many mismatches between query sequence and ATOM record in ', pdbfile)
        exit(1)

    if numMatches < min(30, 0.5 * len(sequence)):
        print('ERROR: more than half of query sequence not covered by ATOM record in ', pdbfile)
        exit(1)

    protein['pdbseq'] = pdbseq
    protein['numMisMatches'] = numMisMatches
    protein['numMatches'] = numMatches

    coordInfo, dssp = result

    protein.update(dssp)

    coordinates, numInvalidAtoms = coordInfo
    if 'CA' in numInvalidAtoms and numInvalidAtoms['CA'] > 10:
        print('ERROR: too many Ca atoms do not have valid 3D coordinates in ', pdbfile)
        exit(1)
    if 'CB' in numInvalidAtoms and numInvalidAtoms['CB'] > 10:
        print('ERROR: too many Cb atoms do not have valid 3D coordinates in ', pdbfile)
        exit(1)

    protein['missing'] = [c is None or (c['CA'] is None and c['CB'] is None) for c in coordinates]

    distMatrix = PDBUtils.CalcDistMatrix(coordinates)
    protein['atomDistMatrix'] = PDBUtils.PostProcessDistMatrix(sequence, distMatrix)

    oriMatrix = PDBUtils.CalcTwoROriMatrix(coordinates)
    protein['atomOrientationMatrix'] = oriMatrix

    oriMatrix = PDBUtils.CalcCaOriMatrix(coordinates)
    protein['atomOrientationMatrix'].update(oriMatrix)
    savefile = os.path.join(resDir, protein['name'] + '.native')
    np.save(savefile, protein)

    if pdbfileIsTemporary:
        os.remove(pdbfile)

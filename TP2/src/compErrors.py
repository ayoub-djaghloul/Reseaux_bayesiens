
#!/usr/bin/python
# coding: utf-8

from optparse import OptionParser
import os
import re
import sys
import codecs


def compErrors(refFile, hypFile):
    
    # keep errors by tags
    dicCount = {}
    dicErrors = {}
    
    refin = codecs.open(refFile, 'r', 'utf8')
    hypin = codecs.open(hypFile, 'r', 'utf8')
    stop = False
    while not stop:
        refLine = refin.readline()
        hypLine = hypin.readline()

        refValues = refLine.strip().split('\t')
        hypValues = hypLine.strip().split('\t')
        # skip empty lines
        while len(refValues) <2 and refLine:
            refLine = refin.readline()
            refValues = refLine.strip().split('\t')
        # skip empty lines
        while len(hypValues) <2 and hypLine:
            hypLine = hypin.readline()
            hypValues = hypLine.strip().split('\t')
      
        if not refLine or not hypLine:
            stop = True
        else:
            # formats: word pos ...
            if refValues[0] != hypValues[0]:
                sys.stderr.write('misaligned files: '+refValues[0]+' != '+hypValues[0])
                sys.exit(1)
            refPOS = refValues[1]
            hypPOS = hypValues[1]
            if refPOS in dicCount:
                dicCount[refPOS] += 1
            else:
                dicCount[refPOS] = 1
                dicErrors[refPOS] = 0
            if refPOS != hypPOS:
                dicErrors[refPOS] += 1

    refin.close()
    hypin.close()
    
    print("Errors: {:d}/{:d} ({:3.2f}%)".format(sum(dicErrors.values()), sum(dicCount.values()),
        (sum(dicErrors.values())/sum(dicCount.values())*100)))
    print('-------------------------')
    print('|     ERRORS BY STATE    |')
    for pos in sorted(dicCount.keys()):
        print('| {:5s}: {:4d} ({:06.2f}%) |'.format(pos, dicErrors[pos], dicErrors[pos]/dicCount[pos]*100))
    print('-------------------------')
   


def main():	
    util = "use : %prog [options] ref hyp\n"+\
           "\tcompare POS tags between both input files\n"

    parser = OptionParser(util)
    
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("incorrect number of arguments"+"\n")

    compErrors(args[0], args[1])
            

if __name__ == "__main__":
	main()

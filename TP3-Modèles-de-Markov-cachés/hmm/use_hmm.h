#ifndef PARSER_H // include guard
#define PERSER_H


#include <string>
#include <map>
#include <vector>
#include <queue>
#include "model.h"

using namespace std;

/**
 * Constant for out-of-vocabulary observations
 */
const string UNK = "UNK";
/**
 * Constant integer: below this count threshold in a data file, an observation is ignored
 * and considered as unkown
 */
const int UNK_THRESHOLD = 2;
/**
 * Special string for observations and hidden states associated with an end of sentance
 * (used with the EOS option of the program)
 */
const string EOS = "EOS";

namespace hmm {
    class Parser
    {
         public:

            Parser()
            { }

            /**
             * List of sequences of observations seen in a file
             * Observations are stored as index numbers
             */
            vector<vector<int>*>* obs_sequences;
            
            /**
             * List of sequences of states seen in a file. Normally, vectors must have the same size
             * as obs_sequences.
             * Observations are stored as index numbers
             */
            vector<vector<int>*>* state_sequences;
            
            /**
             * Reads a training data file to fill obs_sequences and state_sequences and initialize an HMM model.
             * @param filename Name of the input file
             * @param optEOS 'true' if a special observation is added for end of sequences; 'false' otherwise
             * @return initialized HMM (NB: parameters are still not learned)
             */ 
            Model* read_train_data(string filename, bool optEOS);

            /**
             * Reads test data and an HMM model, the last one to associate each observation or hidden state with
             * an index. The objective of this method est to fill obs_sequences and state_sequences.
             * @param filename Name of the input file
             * @param m Trained HMM model
             * @return a FIFO list of observation occurences that are not in the HMM's vocabulary, in the order
             * ot their appearance in the input file
             */
            queue<string>* read_test_data(string filename, Model* m);
            
            /**
             * Writes the sequences of observations with their hidden state found by the HMM model.
             * @param filename Name of the output file
             * @param m Trained HMM model
             * @param unk_obs FIFO list of the observations out of the HMM's vocabulary 
             */
            void write_predict(string filename, Model* m, queue<string>* unk_obs);


         private :

            /**
             * Makes a first read of a data file to get the vocabulary of observations and potentially of 
             * hidden states.
             * @param filename Name of the input file
             * @param optEOS 'true' if a special observation is added for end of sequences; 'false' otherwise
             * @return initialized HMM (NB: parameters are still not learned)
             */
            Model* read_vocab(string filename, bool optEOS);
            
            /**
             * Reads sequences from a file
             * @param filename Name of the output file
             * @param m Trained HMM model
             * @param train 'true' when reading training data; 'false' for test data
             * @return a FIFO list of observation occurences that are not in the HMM's vocabulary, in the order
             * ot their appearance in the input file
             */
            queue<string>* read_sequences(string filename, Model* m, bool train);

    };

}

#endif
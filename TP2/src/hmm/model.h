#ifndef MODEL_H // include guard
#define MODEL_H

#include <vector>
#include <map>
#include <string>


using namespace std;

/**
 * Stop criterion for the Baum-Welch training algorithm
 */
const float STOP_CRITERION = 1e-5;
/**
 * Constant defined for zero proabability to avoid log(-infty) and NaN
 */
const double LOG_MINUS_INFTY = -100;

namespace hmm {
    class Model
    {
        public:
            Model(int nbStates, int vocabSize) 
                : n(nbStates), M(vocabSize)
            {    
                A = new double*[n];
                for(int i = 0; i < n; ++i)
                    A[i] = new double[n];
                
                B = new double*[n];
                for(int i = 0; i < n; ++i)
                    B[i] = new double[M];

                pi = new double[n];

                dic_state = new map<string, int>();
                dic_obs = new map<string, int>();

                tab_state = new string[n];
                tab_obs = new string[M];

            }

            /**
             * Computes posterior probabilities P(Q|O) for a state sequence
             * and a sequence of observations, using alpha and beta (i.e. forward-backward algorithm)
             * @param seq_obs Vector of indices associated withe one sequence of observations
             * @param seq_state Vector of indices associated withe one sequence of hidden states found
             * by the HMM model
             * @return The posterior log-probabilities of each hidden state P(q_t=s_t|O) given by the HMM
             */
            vector<double>* posterior(vector<int>* seq_obs, vector<int>* seq_state);

            /**
             * Predicts the best hidden state sequences for several sequences using the Viterbi algorithm
             * (see the private viterbi() method)
             * @param data_obs Vector of observation sequences
             * @return vector of hidden state sequences found by the HMM model
             */
            vector<vector<int>*>* predict(vector<vector<int>*>* data_obs);

            /**
             * Initializes an HMM from a parameter file
             * NB: parameters are loaded as probabilities and not as log-probabilities 
             * @param filename Name of an input file
             * @return a new HMM object
             */
            static Model* load_hmm(string filename);
            
            /**
             * Saves parameters of an HMM
             * NB: parameters are saved as probabilities and not as log-probabilities 
             * @param filename Name of an output file
             */
            void save_hmm(string filename);

            /** 
             * Trains the model parameters with the unsupervised Baum-Welch algorithm
             * @param data_obs Vector of observation sequences
             * @param filename Saving file for the trained parameters of the HMM
             */
            void train_baum_welch(vector<vector<int>*>* data_obs, string filename);

            /** 
             * Trains the model parameters with supervised data using the MLE criterion 
             * @param data_obs Vector of observation sequences
             * @param data_state Vector of hidden state sequences
             * @param filename Saving file for the trained parameters of the HMM
             */
            void train_MLE(vector<vector<int>*>* data_obs, vector<vector<int>*>* data_state, string filename);

            /**
             * Matrix of transition probabilities
             * A[i][j] = P(q_{t+1}=s_j |q_t=s_i)
             * NB: stores log probabilities
             */
            double** A;

            /**
             * Matrix of observation probabilities
             * B[j][k] = P(O_t=v_k|q_t=s_j)
             * NB: stores log probabilities
            */
            double** B;

            /**
             * Array of initial probabilities
             * pi[i] = P(q_1=s_i)
             * NB: stores log probabilities
             */
            double* pi;

            /**
             * Number of states
             */
            int n;
            /**
             * Size of the vocabulary of observed symbols
             */
            int M;
     
            /**
             * Dictionary of states: dic[state] gives the index of a state
             */
            map<string, int>* dic_state ;

            /**
             * Dictionary of observations: dic[obs] gives the index of an observation
             */
            map<string, int>* dic_obs;
            
            /**
             * Array of string of observations: tab_obs[i]=obs means that obs has i index
             */
            string* tab_obs;
            
            /**
             * Array of strings of stages: tab_state[i]=state means taht state has i
             * index
             */
            string* tab_state;

        private:
           /**
            * Initializes parameters randomly
            * NB: initial probabilities of A MUST be different to allow the Baum-Welch to have several 
            * iterations
            */
            void rand_init_param();

            /**
             * Initializes parameters for the EOS state symbol
             * P(O_t=EOS|q_t=EOS) = 1 and for other observations P(O_t=o|q_t=EOS) = 0
             */
            void init_param_EOS();

            /**
             * Finds the best hidden state sequence Q which maximizes P(Q,O)
             * @param seq_obs Vector of indices associated withe one sequence of observations
             * @return The best hidden states according to the HMM for each observation 
             */
            vector<int>* viterbi(vector<int>* seq_obs);

            /**
             * Computes forward log-probabilities
             * @param seq_obs Vector of indices associated withe one sequence of observations
             * @return Matrix of log-probabilities alpha where alpha[t][i] gives the log-prob of observing
             * O(1..t) and ending at state q_i at time t
             */
            double** alpha(vector<int>* seq_obs);

            /**
             * Computes backward log-probabilities
             * @param seq_obs Vector of indices associated withe one sequence of observations
             * @return Matrix of log-probabilities beta where beta[t][i] gives the log-prob of observing
             * O(t+1..T) and startint at state q_i at time t
             */
            double** beta(vector<int>* seq_obs);

            /** 
             * Computes the log-probability that the model generates a sequence of observations 
             * @param seq_obs Vector of indices associated withe one sequence of observations
             * @param alpha Matrix of forward log-proabilities
             * @return log-probability P(O(1..T)) that the HMM emits O(1..T) 
             */
            double pgen(vector<int>* seq_obs, double** alpha);

            /**
             * Computes the xi log-probilities
             * @param seq_obs Vector of indices associated withe one sequence of observations
             * @param likelihood Variable filled with the likelihood of the sequence, once xi is learned
             * @return Matrix xi where xi[t][i][j] the log-probabliity that s_i emits o_t and s_j 
             * emits o_{t+1}
             */
            double*** xi(vector<int>* seq_obs, double* likelihood);

             /**
              * Computes one iteration of the Baum-Welch algorithm
              * @param data_obs Vector of observation sequences
              * @return log likelihood of the data computed by the HMM in its state at the start of the EM 
              * iteration
              */
            double baum_welch_EMstep(vector<vector<int>*>* data);

            /**
             * Computes the log probabilitiy of the sum of probabilities using the following formula
             * to prevent -infinity log-probabilities when summing very low probabilities:
             * ln(x+y) = ln(x)+ln(1+e^(ln(y)-ln(x)))
             * @param ln_val Array of log-probabilities to sum
             * @param start Start index
             * @param end End index
             * @return ln(sum_{i=start to end} e^ln_val[i]), that is ln(sum_{i=start to end} val[i])
             */
            double ln_of_sum(double* ln_val, int start, int end);

            /**
             * Computes the log probability of the sum of two probabilities using the following formula
             * to prevent -infinity log-probabilities when summing very low probabilities:
             * ln(x+y) = ln(x)+ln(1+e^(ln(y)-ln(x)))
             * @param ln_val1 Log-probability 1
             * @param ln_val2 Log-probabliti 2
             * @return ln(e^ln_val1 + e^ln_val2), that is ln(val1 + val2)
             */
            double ln_of_sum(double ln_val1, double ln_val2);
    };
}

#endif

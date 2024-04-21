#include <iostream>
#include <map>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include "math.h"
#include "use_hmm.h"


using namespace std;
using namespace hmm;




Model* Parser::read_train_data(string filemane, bool optEOS)
{
    Model* model = Parser::read_vocab(filemane, optEOS);
    Parser::read_sequences(filemane, model, true);

    return model;
}

queue<string>* Parser::read_test_data(string filenameIn, Model* m)
{
    queue<string>* res = Parser::read_sequences(filenameIn, m, false);
    state_sequences = m->predict(obs_sequences);
    return res;
}

Model* Parser::read_vocab(string filename, bool optEOS)
{
    Model* model = NULL;
    ifstream filin(filename);
    map<string,int> count_obs;
    map<string,int> count_state ;

    if (filin.is_open())
    {
        string line;
        while ( getline (filin,line) )
        {
            string obs;
            string state ;
            if (!line.empty()) 
            {
                obs = line.substr(0, line.find('\t'));
                count_obs[obs]++;
                state  = line.substr(line.find('\t')+1);
                count_state [state]++;
            }
        }
        filin.close();

           // remove observations below a threshold and associate them with UNK
            map<string, int>::iterator it = count_obs.begin();
            while (it != count_obs.end())
            {
                if (it->second < UNK_THRESHOLD)
                {
                    //cout << "removing: " << it->first << endl;
                    count_obs[UNK]++;
                    count_obs.erase(it++);
                }
                else
                    it++;
            }
        if (optEOS)
        {
            count_obs[EOS] = 1;
            count_state[EOS] = 1;
        }
        // Associate each obs/state  with an index
        int nbStates = count_state.size();
        int vocabSize = count_obs.size();
        cout << "nbStates=" << nbStates <<", vocabSize=" << vocabSize << endl;
        model = new Model(nbStates, vocabSize);

        it = count_obs.begin();
        int index = 0;
        while (it != count_obs.end())
        {
            model->tab_obs[index] = it->first;
            (*model->dic_obs)[it->first] = index;
            index++;
            it++;
        } 
        
        it = count_state .begin();
        index = 0;
        while (it != count_state .end())
        {
            model->tab_state[index] = it->first;
            (*model->dic_state )[it->first] = index;
            index++;
            it++;
        }

    }

    else cerr << "Unable to open file: " << filename << endl;
    return model;
}

queue<string>* Parser::read_sequences(string filename, Model* model, bool train)
{
    queue<string>* unk_obs = new queue<string>();

    ifstream filin(filename);
    obs_sequences = new vector<vector<int>*>();
    state_sequences = new vector<vector<int>*>();
    

    if (filin.is_open())
    {
        string line;
        vector<int>* obs_seq1 = new vector<int>();
        vector<int>* state_seq1 = new vector<int>();
        while ( getline (filin,line) )
        {
            string obs;
            string state ;
            if (line.empty()) 
            {
                if (!obs_seq1->empty())
                {
                    if(model->dic_obs->count(EOS) > 0)
                    {
                        obs_seq1->push_back((*model->dic_obs)[EOS]);
                        if (train)
                            state_seq1->push_back((*model->dic_state)[EOS]);
                    }

                    obs_sequences->push_back(obs_seq1);
                    if (train)
                        state_sequences->push_back(state_seq1);

                    // new sentence
                    obs_seq1 = new vector<int>();
                    if (train)
                        state_seq1 = new vector<int>();
                }
                
             } else {
                if (train)
                {
                    obs = line.substr(0, line.find('\t'));
                    state  = line.substr(line.find('\t')+1);
                    if (model->dic_obs->count(obs) > 0)
                        obs_seq1->push_back((*model->dic_obs)[obs]);
                    else
                    {
                        obs_seq1->push_back((*model->dic_obs)[UNK]);
                        // useless for train: unk_obs->push(obs);
                    }
                    state_seq1->push_back((*model->dic_state)[state]);
                } else {
                    obs = line;
                    if (model->dic_obs->count(obs) == 0)
                    {
                        if (model->dic_obs->count(UNK) == 0)
                        {
                            cerr << "the observation: " << obs << " has not been previously seen by the model and the model cannot manage unknown observations" <<endl;
                            exit(1);
                        }
                        else
                        {
                            obs_seq1->push_back((*model->dic_obs)[UNK]);
                            unk_obs->push(obs);
                        }                        
                    } 
                    else
                        obs_seq1->push_back((*model->dic_obs)[obs]);
                }
            }
  
        }
        if (!obs_seq1->empty())
        {
            if(model->dic_obs->count(EOS) > 0)
            {
                obs_seq1->push_back((*model->dic_obs)[EOS]);
                if (train)
                    state_seq1->push_back((*model->dic_state)[EOS]);  
            }

            obs_sequences->push_back(obs_seq1);
            if (train)
            {
                state_sequences->push_back(state_seq1);
            }
        }
        filin.close();
    }
 
    else cerr << "Unable to open file: " << filename << endl;
   return unk_obs;
}

void Parser::write_predict(string filename, Model* model, queue<string>* unk_obs) 
{
    if (obs_sequences->size()==state_sequences->size())
    {
        ofstream filout;
        filout.open(filename);
        for (int k=0; k<obs_sequences->size(); k++)
        {
                
            vector<int>* obs_seq1 = (*obs_sequences)[k];
            vector<int>* state_seq1 = (*state_sequences)[k];
            vector<double>* posteriors = model->posterior(obs_seq1, state_seq1);

            bool hasUNK =  model->dic_obs->count(UNK) > 0;
            bool hasEOS =  model->dic_obs->count(EOS) > 0;
            // Assertion: obs_seq1, state_seq_1 and posteriors have the same size
            for (int i=0; i<obs_seq1->size(); i++)
            {
                if (hasEOS && (*obs_seq1)[i] == (*model->dic_obs)[EOS])
                {
                    // remove EOS in output
                }
                else 
                {
                    if (hasUNK && (*obs_seq1)[i] == (*model->dic_obs)[UNK])
                    {
                        filout << unk_obs->front();
                        unk_obs->pop();
                    }
                    else
                    {
                        filout << model->tab_obs[(*obs_seq1)[i]];
                    }
                    filout << "\t" << model->tab_state[(*state_seq1)[i]] << "\t" << exp((*posteriors)[i]) << endl;
                    // filout << "\t" << model->tab_state[(*state_seq1)[i]] << endl;
                }
            }
            filout << endl;
        }
        filout.close();
    }
    else cerr << "Error in the result of Viterbi outputs" << endl;
}

int main(int argc, char *argv[])
{
    Parser parser;
    if (argc > 3) 
    {
        if (string(argv[1]) == "train_unsup")
        {
            bool optEOS = false;
            if (argc > 4 && strcmp(argv[4],"EOS")==0)
            {
                optEOS = true;    
            }
            Model* m = parser.read_train_data(argv[3], optEOS);
            m->train_baum_welch(parser.obs_sequences, argv[2]);
        } 
        else if (string(argv[1]) == "train_sup")
        {
            bool optEOS = false;
            if (argc > 4 && strcmp(argv[4],"EOS")==0)
            {
                optEOS = true;
            }
            Model* m = parser.read_train_data(argv[3], optEOS);
            m->train_MLE(parser.obs_sequences, parser.state_sequences, argv[2]);
        }  
        else if (string(argv[1]) == "test")
        {
            Model* m = Model::load_hmm(argv[2]);
            queue<string>* unk_obs = parser.read_test_data(argv[3], m);
            parser.write_predict(argv[4], m, unk_obs);
        }
    } else
    {
        cerr << "you need to specify at least 3 arguments to this program\n" <<
        "\tFormat: %prog {train_sup,train_unsup} <model_file> <training_data> (EOS)?\n" <<
        "\t        %prog test <model_file> <test_data> <output_data>"<< endl;
    }
    
}
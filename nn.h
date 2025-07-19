
/* Total number of neural internal hidden layer which excludes
   Input layer and output layer. So if NUM_OF_INTERNAL_HIDDEN_LAYERS is 2
   then neural network would look like

   0
   0        0               0
   0        0       0       0
   0        0       0       0
   0        0       0       0
   0        0       0       0
   0        0       0       0
   0        0               0
   0
   I/P     Hid     Hid     O/P
           Lay1    Lay2    Lay
   Note: Number of nodes in all the layers are arbitrary.
*/
#define NUM_OF_INTERNAL_HIDDEN_LAYERS       (2)
/* Number of nodes in individual hidden layers */
static const unsigned int nn_hid_lay_nodes[NUM_OF_INTERNAL_HIDDEN_LAYERS] = {
                                                                      30 /* Node in Layer 1 */,
                                                                      24 /* Node in Layer 2 */
                                                                      };

typedef struct
{
    float *w; /*All weights*/
    float *hw[NUM_OF_INTERNAL_HIDDEN_LAYERS]; /*hidden layer to output layer weights*/
    float *b; /*biases*/
    float *h[NUM_OF_INTERNAL_HIDDEN_LAYERS]; /*hidden layer*/
    float *o; /*output layer*/
    int nb;   /*number of biases*/
    int nw;   /*number of weights*/
    int nips; /*number of inputs*/
    int nhid[NUM_OF_INTERNAL_HIDDEN_LAYERS]; /*number of hidden neurons*/
    int nops; /*number of outputs*/
}NeuralNetwork_Type;

float * NNpredict(const NeuralNetwork_Type nn, const float * in );
NeuralNetwork_Type NNbuild(const int nips, const int nops);
float NNtrain(const NeuralNetwork_Type nn, const float * in,const float * tg,float rate);
void NNsave(const NeuralNetwork_Type nn, const char * path);
NeuralNetwork_Type NNload(const char * path);
void NNprint(const float * arr, const int size);
void NNfree(const NeuralNetwork_Type nn);

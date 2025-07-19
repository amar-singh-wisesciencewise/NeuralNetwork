#include "nn.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>


static float act(const float a);
static float pdact(const float a );
static float err(const float a, const float b);
static float toterr(const float * const tg,const float * const o, const int size);
static float pderr(const float a, const float b);
static float frand();


static void fprop(const NeuralNetwork_Type nn,const float * const in){
   /*Hidden layer 1 neuron values*/
   for(int i = 0; i < nn.nhid[0]; i++){
        float sum = 0.0f;
        for(int j = 0; j < nn.nips; j++){
            sum += in[j] * nn.w[i * nn.nips + j];
        }
        nn.h[0][i] =  act(sum + nn.b[0]);
   }

   /* Hidden layer2 values */
   for(int i=0; i < nn.nhid[1]; i++){
        float sum = 0.0f;
        for(int j=0; j < nn.nhid[0]; j++){
            sum += nn.h[0][j] * nn.hw[0][i * nn.nhid[0] + j];
        }
        nn.h[1][i] =  act(sum + nn.b[1]);
   }

    /*Output layer neuron values*/
    for(int i = 0; i < nn.nops; i++){
        float sum  = 0.0f;
        for(int j = 0; j < nn.nhid[1]; j++){
            sum += nn.h[1][j] * nn.hw[1][i * nn.nhid[1] + j];
        }
        nn.o[i] = act(sum + nn.b[2]);
    }
}

static void bprop(const NeuralNetwork_Type nn,
                  const float *const in,
                  const float * const tg,
                  float rate)
{
    float *err_hid = (float *)calloc(nn.nhid[0], sizeof(*err_hid));

    for(int i = 0; i < nn.nhid[1]; i++){
        float pd_hid = 0.0f;
        for(int j = 0; j < nn.nops; j++){
          const float a = pderr(nn.o[j],tg[j]);
          const float b = pdact(nn.o[j]);

          pd_hid += a * b * nn.hw[1][j * nn.nhid[1] + i];

          nn.hw[1][j * nn.nhid[1] + i] -= rate * a * b * nn.h[1][i];
        }

        for(int j = 0; j < nn.nhid[0]; j++){
            err_hid[j] += pd_hid * pdact(nn.h[1][i]) * nn.hw[0][i * nn.nhid[0] + j];
            nn.hw[0][i * nn.nhid[0] + j] -= rate * pd_hid * pdact(nn.h[1][i]) * nn.h[0][j];
        }
    }
    for(int j = 0; j < nn.nhid[0]; j++){
        for(int k = 0; k < nn.nips; k++){
            nn.w[j * nn.nips + k] -= rate * err_hid[j] * pdact(nn.h[0][j])* in[k];
        }
    }

    free(err_hid);
}

static  void wbrand(const NeuralNetwork_Type nn)
{
     for(int i = 0; i< nn.nw; i++){
        nn.w[i] = frand() -0.5f;
     }

     for(int i=0; i < nn.nb; i++){
        nn.b[i] = frand() - 0.5f;
     }
}

float * NNpredict(const NeuralNetwork_Type nn, const float * in ){
   fprop(nn,in);
   return nn.o;
}

NeuralNetwork_Type NNbuild(const int nips, const int nops){
   NeuralNetwork_Type nn;
   /* TODO: Needs to make bias per node in a layer instead of one per layer */
   nn.nb = NUM_OF_INTERNAL_HIDDEN_LAYERS + 1;  /*number of biases*/
   /* TODO: Generalize and use loop so that arbitrary loops can be supported */
   nn.nw =  nips * nn_hid_lay_nodes[0] +
            nn_hid_lay_nodes[0] * nn_hid_lay_nodes[1] +
            nn_hid_lay_nodes[1] * nops; /*number of  total weights*/
    /* Weigths of all the layesr
        Start will have weights for I/P and Hidden-Layer-1
        Then weights for Hidden-Layer-1 and Hidden-Layer-2
        In last, Weights for Hidden-Layer-2 and O/P layer
    */
   nn.w =  (float *)calloc(nn.nw,sizeof(*nn.w));  /*All weights*/
   nn.hw[0] =  nn.w + nn_hid_lay_nodes[0] * nips;  /* hidden layer1 to hidden layer2 weights */
   nn.hw[1] =  nn.hw[0] + nn_hid_lay_nodes[0] * nn_hid_lay_nodes[1];   /* hidden layer2 to output layer weights*/

   nn.b = (float *)calloc(nn.nb,sizeof(*nn.b));   /*biases*/
   nn.h[0] = (float *)calloc(nn_hid_lay_nodes[0],sizeof(*nn.h[0])); /*hidden layer 1*/
   nn.h[1] = (float *)calloc(nn_hid_lay_nodes[1],sizeof(*nn.h[1])); /* Hiddent Layer 2 */
   nn.o = (float *)calloc(nops,sizeof(*nn.o));    /*output layer*/
   nn.nips = nips;                              /*number of inputs*/
   nn.nhid[0] =  nn_hid_lay_nodes[0];           /*number of hidden neurons*/
   nn.nhid[1] =  nn_hid_lay_nodes[1];           /*number of hidden neurons*/
   nn.nops = nops;                              /*number of outputs*/
   wbrand(nn);

   return nn;
}

void NNsave(const NeuralNetwork_Type nn, const char * path){
 FILE * const file =  fopen(path,"w");
 /*Save the header*/
 fprintf(file,"%d %d %d %d\n",nn.nips,nn.nhid[0],nn.nhid[1],nn.nops);

  /*Save the biases*/
 for(int i =0;i<nn.nb;i++){
    fprintf(file,"%f\n",(double)nn.b[i]);
 }
  /*Save the weights*/
 for(int i=0;i<nn.nw;i++){
    fprintf(file,"%f\n",(double)nn.w[i]);
 }
 fclose(file);
}

NeuralNetwork_Type NNload(const char * path){
FILE * const file = fopen(path,"r");
int nips =0;
int nhid[NUM_OF_INTERNAL_HIDDEN_LAYERS] = {};
int nops =0;

 /*Load the header*/
 fscanf(file, "%d %d %d %d\n",&nips,&nhid[0],&nhid[1],&nops);

 const NeuralNetwork_Type nn =  NNbuild(nips,nops);

  /*Load the biases*/
  for(int i=0;i<nn.nb;i++){
    fscanf(file,"%f\n",&nn.b[i]);
  }
   /*Load the weights*/
  for(int i =0;i<nn.nw;i++){

    fscanf(file,"%f\n",&nn.w[i]);
  }

  fclose(file);
return nn;
}


float NNtrain(const NeuralNetwork_Type nn, const float * in,const float * tg,float rate){
    fprop(nn,in);
    bprop(nn,in,tg,rate);

    return toterr(tg,nn.o,nn.nops);
}

void NNprint(const float * arr, const int size){
 double max =  0.0f;
 int idx;

 for(int i=0;i <size;i++){
    printf("%f ",(double)arr[i]);

    if(arr[i] > max){
        idx = i;
        max =  arr[i];
    }
 }
 printf("\n");
 printf("The number is :%d\n",idx);
}
void NNfree(const NeuralNetwork_Type nn){
 free(nn.w);
 free(nn.b);
 free(nn.h[0]);
 free(nn.h[1]);
 free(nn.o);
}


static float err(const float a, const float b){
   return 0.5f*(a-b)*(a-b);
}

static float toterr(const float * const tg,const float * const o, const int size){

  float sum= 0.0f;
  for(int i=0;i<size;i++){

    sum +=err(tg[i],o[i]);
  }
  return sum;
}

static float pderr(const float a, const float b){
   return a - b;
}

static float act(const float a){
    return 1.0f/(1.0f + expf(-a));
}

static float pdact(const float a ){
  return a*(1.0f -a);
}

static float frand(){
  return rand()/(float)RAND_MAX;
}

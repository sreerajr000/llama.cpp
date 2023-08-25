#include <vector>

typedef std::vector<float> tensor1d;
typedef std::vector<tensor1d> tensor2d;
typedef std::vector<tensor2d> tensor3d;

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

struct TransformerWeights {
    tensor2d token_embedding_table;
    tensor2d rms_att_weight;
    tensor2d rms_ffn_weight;
    tensor3d wq;
    tensor3d wk;
    tensor3d wv;
    tensor3d wo;
    tensor3d w1;
    tensor3d w2;
    tensor3d w3;
    tensor1d rms_final_weight;
    tensor1d wcls;
};


struct RunState {
    tensor1d x;
    tensor1d xb;
    tensor1d xb2;
    tensor1d hb;
    tensor1d hb2;
    tensor1d q;
    tensor1d k;
    tensor1d v;
    tensor2d att;
    tensor1d logits;
    tensor3d key_cache;
    tensor3d value_cache;
};

struct Transformer {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
};


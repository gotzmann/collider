#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#if !defined (_WIN32)
#include <stdio.h>
#include <termios.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include "ggml-backend.h"
// #include "common.h"
#include "llama.h"
//#include "sampling.h"
#include "common/grammar-parser.h"

#ifdef _WIN32
#define NULL_DEVICE "NUL:"
#define TTY_DEVICE "COM1:"
#else
#define NULL_DEVICE "/dev/null"
#define TTY_DEVICE "/dev/tty"
#endif

// ------------------------------------------------------

int32_t get_num_physical_cores(); 

// this is a common sampling function used across the examples for convenience
// it can serve as a starting point for implementing your own sampling function
// Note: When using multiple sequences, it is the caller's responsibility to call
//       llama_sampling_reset when a sequence ends
//
// required:
//  - ctx_main:     context to use for sampling
//  - ctx_sampling: sampling-specific context
//
// optional:
//  - ctx_cfg:      context to use for classifier-free guidance
//  - idx:          sample from llama_get_logits_ith(ctx, idx)
//
// returns:
//  - token:      sampled token
//  - candidates: vector of candidate tokens
//
llama_token llama_sampling_sample(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        struct llama_context * ctx_cfg,
        int idx = 0);

void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar);

// Get the last sampled token
llama_token llama_sampling_last(llama_sampling_context * ctx);              

// ------------------------------------------------------

// --- Some (possibly be wrong) observations

// Temp is used both for mirostat and TopK-TopP sampling, but it better to keep lower temp for mirostat

// Mirostat is more chatty and fluid, looks like real human being. At the same time is jumps wild between topics
// It favors bigger models and becomes crazy on smaller ones like 7B
// Mirostat v2 looks better than v1
// Results with tau = 5 norm, better with lower values down to 1.0 and wild bad up to 9.0
// Not sure how eta plays here

// TopK over 40 (up to 100) might produce crazy irrelevant results. But usually it safe to lower about 10-20
// Seems like TopP sweet spot is arount 0.95
// Not sure about Temp, but any value around 0.5 .. 0.8 is good enough (lower == more steady output, higher = more creative)

// RepeatPenalty is good around 1.1 and bad around 1.0 (switched off). 
// Not sure why bad values above 1.1 are shuffling and muffling the output completely (with mirostat at least)

// --- sampling parameters

// sampler types
enum class llama_sampler_type : char {
    TOP_K       = 'k',
    TOP_P       = 'p',
    MIN_P       = 'm',
    TFS_Z       = 'f',
    TYPICAL_P   = 'y',
    TEMPERATURE = 't'
};

// sampling parameters
typedef struct llama_sampling_params {

    // -- Janus Sampling

    int32_t janus = 1;    // 0 = off or Janus Sampling version
    int32_t depth = 200;  // last n tokens to penalize [ -1 = context size ]
    float   scale = 0.96; // janus scale factor for penalty and other heuristics
    float   hi    = 0.99; // 1.0 = max pedantic [ 100% strict ]
    float   lo    = 0.96; // 0.0 = min pedantic [ 100% random ]

    // -- mainstream samplings

    int32_t     n_prev                = 64;       // number of previous tokens to remember
    int32_t     n_probs               = 0;        // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t     min_keep              = 0;        // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t     top_k                 = 40;       // <= 0 to use vocab size
    float       top_p                 = 0.95f;    // 1.0 = disabled
    float       min_p                 = 0.05f;    // 0.0 = disabled
    float       tfs_z                 = 1.00f;    // 1.0 = disabled
    float       typical_p             = 1.00f;    // 1.0 = disabled
    float       temp                  = 0.80f;    // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range        = 0.00f;    // 0.0 = disabled
    float       dynatemp_exponent     = 1.00f;    // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t     penalty_last_n        = 64;       // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat        = 1.10f;    // 1.0 = disabled
    float       penalty_freq          = 0.00f;    // 0.0 = disabled
    float       penalty_present       = 0.00f;    // 0.0 = disabled
    int32_t     mirostat              = 0;        // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau          = 5.00f;    // target entropy
    float       mirostat_eta          = 0.10f;    // learning rate
    bool        penalize_nl           = true;     // consider newlines as a repeatable token

    std::vector<llama_sampler_type> samplers_sequence = {
        llama_sampler_type::TOP_K,
        llama_sampler_type::TFS_Z,
        llama_sampler_type::TYPICAL_P,
        llama_sampler_type::TOP_P,
        llama_sampler_type::MIN_P,
        llama_sampler_type::TEMPERATURE
    };

    std::string grammar;  // optional BNF-like grammar to constrain sampling

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    std::string cfg_negative_prompt; // string to help guidance
    float       cfg_scale     = 1.f; // how strong is guidance

    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

    std::vector<llama_token> penalty_prompt_tokens;
    bool                     use_penalty_prompt_tokens = false;
} llama_sampling_params;

// --- gpt params

struct gpt_params {
    uint32_t seed                 = -1;    // RNG seed

    int32_t n_threads             = get_num_physical_cores();
    int32_t n_threads_draft       = -1;
    int32_t n_threads_batch       = -1;    // number of threads to use for batch processing (-1 = use n_threads)
    int32_t n_threads_batch_draft = -1;
    int32_t n_predict             = -1;    // new tokens to predict
    int32_t n_ctx                 = 512;   // context size
    int32_t n_batch               = 512;   // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                = 0;     // number of tokens to keep from initial prompt
    int32_t n_draft               = 8;     // number of tokens to draft during speculative decoding
    int32_t n_chunks              = -1;    // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel            = 1;     // number of parallel sequences to decode
    int32_t n_sequences           = 1;     // number of sequences to decode
    float   p_accept              = 0.5f;  // speculative decoding accept probability
    float   p_split               = 0.1f;  // speculative decoding split probability
    int32_t n_gpu_layers          = -1;    // number of layers to store in VRAM (-1 - use default)
    int32_t n_gpu_layers_draft    = -1;    // number of layers to store in VRAM for the draft model (-1 - use default)
    llama_split_mode split_mode   = LLAMA_SPLIT_LAYER; // how to split the model across GPUs
    int32_t main_gpu              = 0;     // the GPU that is used for scratch and small tensors
    float   tensor_split[128]     = {0};   // how split tensors should be distributed across GPUs
    int32_t n_beams               = 0;     // if non-zero then use beam search of given width.
    int32_t grp_attn_n            = 1;     // group-attention factor
    int32_t grp_attn_w            = 512;   // group-attention width
    int32_t n_print               = -1;    // print token count every n tokens (-1 = disabled)
    float   rope_freq_base        = 0.0f;  // RoPE base frequency
    float   rope_freq_scale       = 0.0f;  // RoPE frequency scaling factor
    float   yarn_ext_factor       = -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor      = 1.0f;  // YaRN magnitude scaling factor
    float   yarn_beta_fast        = 32.0f; // YaRN low correction dim
    float   yarn_beta_slow        = 1.0f;  // YaRN high correction dim
    int32_t yarn_orig_ctx         = 0;     // YaRN original context length
    int32_t rope_scaling_type     = LLAMA_ROPE_SCALING_UNSPECIFIED;
    ggml_numa_strategy numa       = GGML_NUMA_STRATEGY_DISABLED;

    // // sampling parameters
    struct llama_sampling_params sparams;

    std::string model             = "models/7B/ggml-model-f16.gguf"; // model path
    std::string model_draft       = "";                              // draft model for speculative decoding
    std::string model_alias       = "unknown"; // model alias
    std::string prompt            = "";
    std::string prompt_file       = "";  // store the external prompt file name
    std::string path_prompt_cache = "";  // path to file for saving/loading prompt eval state
    std::string input_prefix      = "";  // string to prefix user inputs with
    std::string input_suffix      = "";  // string to suffix user inputs with
    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted
    std::string logdir            = "";  // directory in which to save YAML log files
    std::string logits_file       = "";  // file for saving *all* logits

    std::vector<llama_model_kv_override> kv_overrides;

    // TODO: avoid tuple, use struct
    std::vector<std::tuple<std::string, float>> lora_adapter; // lora adapter path with user defined scale
    std::string lora_base  = "";                              // base model path for the lora adapter

    int  ppl_stride        = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int  ppl_output_type   = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                    //                                       (which is more convenient to use for plotting)
                                    //
    bool   hellaswag       = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks = 400;   // number of tasks to use when computing the HellaSwag score

    bool   winogrande      = false; // compute Winogrande score over random tasks from datafile supplied in prompt
    size_t winogrande_tasks= 0;     // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice = false; // compute TruthfulQA score over random tasks from datafile supplied in prompt
    size_t multiple_choice_tasks = 0;     // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool   kl_divergence   = false; // compute KL-divergence

    bool mul_mat_q         = true;  // if true, use mul_mat_q kernels instead of cuBLAS
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode
    bool chatml            = false; // chatml mode (used for models trained on chatml syntax)
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool embedding         = false; // get only sentence embedding
    bool escape            = false; // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool interactive_first = false; // wait for user input immediately
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = false; // insert new sequences for decoding on-the-fly

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos        = false; // ignore generated EOS tokens
    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool logits_all        = false; // return logits for all tokens in the batch
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool display_prompt    = true;  // print prompt before generation
    bool infill            = false; // use infill mode
    bool dump_kv_cache     = false; // dump the KV cache contents for debugging purposes
    bool no_kv_offload     = false; // disable KV offloading

    std::string cache_type_k = "f16"; // KV cache data type for the K
    std::string cache_type_v = "f16"; // KV cache data type for the V

    // multimodal models (see examples/llava)
    std::string mmproj = ""; // path to multimodal projector
    std::string image  = ""; // path to an image file
};

// Create a new sampling context instance.
struct llama_sampling_context * llama_sampling_init(const struct llama_sampling_params & params);

// general sampler context
// TODO: move to llama.h
struct llama_sampling_context {
    // parameters that will be used for sampling
    llama_sampling_params params;

    // mirostat sampler state
    float mirostat_mu;

    llama_grammar * grammar;

    // internal
    grammar_parser::parse_state parsed_grammar;

    // TODO: replace with ring-buffer
    std::vector<llama_token>      prev;
    std::vector<llama_token_data> cur;
};


//
// Sampling utils
//

llama_token sample_top_token(/*struct llama_context * ctx,*/ const float * logits, const int size);

// this is a common sampling function used across the examples for convenience
// it can serve as a starting point for implementing your own sampling function
//
// required:
//  - ctx:    context to use for sampling
//  - params: sampling parameters
//
// optional:
//  - ctx_guidance:  context to use for classifier-free guidance, ignore if NULL
//  - grammar:       grammar to use for sampling, ignore if NULL
//  - last_tokens:   needed for repetition penalty, ignore if empty
//  - idx:           sample from llama_get_logits(ctx) + idx * n_vocab
//
// returns:
//  - token:      sampled token
//  - candidates: vector of candidate tokens
//
llama_token llama_sample_token(
                struct llama_context * ctx,
                struct llama_context * ctx_guidance,
                struct llama_grammar * grammar,
               llama_sampling_params & params,
      const std::vector<llama_token> & last_tokens,
       std::vector<llama_token_data> & candidates,
                        const size_t   promptLen,
                        const size_t   pos, 
                        const size_t   max);

std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos,
                        bool   special);

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_bos,
                        bool   special);   

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token);  

// Uses the value from the model metadata if possible, otherwise
// defaults to true when model type is SPM, otherwise false.
bool llama_should_add_bos_token(const llama_model * model); 

/*
std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_bos,
                        bool   special = false);

std::string llama_token_to_str(const struct llama_context * ctx, llama_token token);
*/

void hide();
void show();

struct llama_context * init_context(int idx);
int64_t do_inference(
    int idx, 
    struct llama_context * ctx, 
    const std::string & jobID, 
    const std::string & sessionID, 
    const std::string & text);

const char * statusCPP(const std::string & jobID);
int64_t promptEvalCPP(const std::string & jobID);
int64_t getPromptTokenCountCPP(const std::string & jobID);
int64_t timingCPP(const std::string & jobID);
uint32_t getSeedCPP(const std::string & jobID);

extern "C" { // -----    

void init(char * swap, char * debug);

void * initContext(
    int idx, 
    char * modelName, 
    int threads,
    int batch_size,
    int gpu1, int gpu2, 
    int context, int predict,
    int32_t mirostat, float mirostat_tau, float mirostat_eta,
    float temperature, int top_k, float top_p,
    float typical_p,
    float repetition_penalty, int penalty_last_n,
    int32_t janus,
	int32_t depth,
	float scale,
	float hi,
	float lo,
    uint32_t seed,
    char * debug);

int64_t doInference(
    int idx, 
    void * ctx, 
    char * jobID, 
    char * sessionID, 
    char * prompt); 

void stopInference(int idx);
const char * status(char * jobID);
int64_t promptEval(char * jobID);
int64_t getPromptTokenCount(char * jobID);
int64_t timing(char * jobID);  
uint32_t getSeed(char * jobID);  

} // ------- extern "C"

// For internal test use
const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(struct llama_context * ctx);

std::vector<std::byte> getBytes(std::string const &s);
bool isPedantic(llama_token id);
int tokType(const llama_context *ctx, const llama_token token);
int tokSize(const llama_context *ctx, const llama_token token);

// Batch utils

void llama_batch_clear(struct llama_batch & batch);

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

## llama3.c - A faithful clone of Karpathy's llama2.c but fully functional with LLaMA 3 8B base and instruct models.

See [Andrej Karpathy's repo](https://github.com/karpathy/llama2.c) for the real deal built for llama2.c architecture and many other cool models he has built.

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Have you ever wanted to inference a baby [Llama 3](https://ai.meta.com/llama/) model in pure C? No? Well, now you can!

Run LLaMA 3 8B models with one simple 700-line C file ([run.c](run.c)). 

The current code inferences models in both fp32 and int8 (see below).

Please note that this repo is a modificaion of Andrej Karpathy's llama2.c but changing the hard coding to work with the modified-tiktoken tokenization used by the suite of Meta LLaMA 3 models.

## getting started

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/jameswdelancey/llama3.c.git
```

Then, open the repository folder:

```bash
cd llama3.c
```

## which model do i download? base or instruct
- If you do not know, go with the instruct model. It will work with both the single shot "generate" mode and the "chat" mode of llama3.c.
- The "chat" mode of llama3.c only supports the instruct model and will surely not work with base model. You can try it for fun and learning at your own risk :).

Download LLaMA 3 8B base and/or instruct. The huggingface site works. You'll need to sign up and get approved.
Specifically download the `original` directory.

```
https://huggingface.co/meta-llama/Meta-Llama-3-8B
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

When downloading these models I did have to rename the original-params.json to params.json for the export.py to work.

```
mv /d/llama3-8b-instruct/original_params.json /d/llama3-8b-instruct/params.json
```

# compile and run the C code:

```bash
gcc -Ofast run.c -o run
./run.exe "llama3_8b_instruct.bin" -z "tokenizer_llama3.bin" -m chat
./run.exe "llama3_8b_instruct.bin" -z "../dev/tokenizer_llama3.bin" -i "Once upon a time"
```

# high performance

- fopenmp If you have these libraries you can run the model much faster. I'm running an Intel i3 14th gen and get 1.9 tok/s with openmp
- march=native This is required for gcc or clang to use SIMD intrinsics and will speed up your runs.
- win.c This is optional unless you're on Windows. I'm compiling with MINGW64 and it works well.
- gcc or clang, both work well and I get very close results between the two.

```bash
$ gcc -Ofast -fopenmp -march=native run.c win.c -o run
```

## performance experiments with batched sgemm

Community experiments have shown that linking `run.c` against a high performance
BLAS implementation can dramatically increase both prompt processing and token
generation throughput on CPU-only systems. In particular, a fork that wires
Intel MKL 2023.1.0 into the attention and prompt-processing paths via
`cblas_sgemm_batched()` reports the following single-threaded results on a Ryzen
7 5800X when using GCC 14.1.1 and the prompt from the example below:

| build | prompt tokens/s | generated tokens/s |
|-------|-----------------|--------------------|
| baseline `run` | ~1.0 | ~1.0 |
| MKL fork | 53.70 | 1.33 |
| llama.cpp (for reference) | 46.14 | 1.29 |

The gains primarily come from:

- A dedicated prompt-processing routine that parallelizes per-token matrix work
  with batched GEMM calls.
- Rewriting the attention path to rely on `cblas_sgemm_batched()`.
- Avoiding redundant computations inside the transformer loop and measuring the
  full prompt window instead of skipping the first token.

The fork also provides experimental OpenBLAS and standalone back ends. However,
the OpenBLAS path currently regresses token generation throughput unless
`forward()` falls back to the original `matmul()` routine, and the standalone
path lacks an optimized SGEMM kernel. Further work on custom SGEMM
implementations could enable additional optimizations such as fusing softmax
preparation into the matrix multiply.

# base model single shot
This still runs at interactive rates and samples more coherent and diverse stories:

```bash
./run "llama3_8b_base.bin" -z "../dev/tokenizer_llama3.bin" -n 50 -i "Once upon a time"
```

> Once upon a time, a girl named Lily who had grown up in a world where people werenΓÇÖt allowed to live happily ever after. One day, Lily decided to take a chance and do something that was a little bit crazy, something that she had

You can also prompt the model with a prefix or a number of additional command line arguments, e.g. to sample at temperature 0.8 for 256 steps and with a prompt:

```bash
./run "llama3_8b_base.bin" -z "../dev/tokenizer_llama3.bin" -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
```

> One day, Lily met a Shoggoth named Lurking. She had never seen a Shoggoth before, but she was immediately impressed by its size and ferocity. They struck up a conversation, and Lurking told her about the ShoggothsΓÇÖ origins and their place in the universe. Lily was fascinated by what Lurking told her, and she learned a lot about the Shoggoths that day. She also made a new friend in Lurking.

# instruct model, chat

You're very likely to get results that you like better from the instruct model. You must use it with the chat mode as the token setup (preamble) is totally different. Here's an example:

```bash
./run "llama3_8b_instruct.bin" -z "../dev/tokenizer_llama3.bin" -m chat
```

> Enter system prompt (optional): You are a professional software programmer with 20 years experience and an IQ of 200 and you love helping people.
> 
> User: Give me a terse hello world in Rust
> 
> Assistant: A delightful request!
>
> Here is a terse "Hello, World!" program in Rust:
> ```rust
> fn main() {
>     println!("Hello, World!");
> }
> ```
> This program defines a `main` function that calls the `println!` macro to print the string "Hello, World!" followed by a newline character.
>
> That's it! You can run this program using the `rustc` compiler and then executing the resulting executable file.
> 
> User:

## Meta's Llama 3 models

Sadly there is a bit of friction here due to licensing (I can't directly upload the checkpoints, I think). So Step 1, get the Llama 3 checkpoints by following the [Meta instructions](https://github.com/facebookresearch/llama3). Once we have those checkpoints, we have to convert them into the llama3.c format.
For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export.py` file, e.g. for 8B model:

```bash
python export.py llama3_8b_base.bin --meta-llama ../llama3-8b-base/
```

The export will take ~10 minutes or so and generate a 31GB file (the weights of the 8B model in float32) called `llama3_8b.bin` in the current directory. Once the export is done, we can run it:

```bash
./run "llama3_8b_base.bin" -z "../dev/tokenizer_llama3.bin"
```

This ran at about 2 tokens/s compiled with [OpenMP](#OpenMP) on 8 threads on my Intel i3 14th gen. Example output:

```bash
./run "llama3_8b_base.bin" -z "../dev/tokenizer_llama3.bin"
```
> Question:
> What is the second derivative of 2*p**3 - 4*p**2 - 12*p?
> Answer:
> 12*p - 8

base models... ¯\\_(ツ)_/¯. Since we can inference the base model, it should be possible to also inference the instruct model quite easily, and have a conversation with it. And if we can find a way to run 7B more efficiently, we can start adding LoRA to our training script, and going wild with finetunes all within the repo!

You can also chat with the Llama Chat models. Export the chat model exactly as above:

```bash
python export.py llama3_8b_instruct.bin --meta-llama ../llama3-8b-instruct/
```

Then chat with it by specifying the chat mode using the `-m` flag, e.g.:

```bash
./run "llama3_8b_instruct.bin" -z "../dev/tokenizer_llama3.bin" -m chat
```
## Int8 Quantization

Compile the quantized version of the runtime:
```bash
gcc -Ofast -fopenmp -march=native runq.c win.c -o runq
```
Export a quantized version of the model. It is about 8GB vs 31.

```bash
python export.py llama3_8b_instruct_q80.bin --meta-llama ../llama3-8b-base/ --version 2
```

The export will take ~10 minutes or so. Once the export is done, we can run it:

```bash
./runq "llama3_8b_instruct_q80.bin" -z "../dev/tokenizer_llama3.bin"
```

This ran at about 4 tokens/s compiled with [OpenMP](#OpenMP) on 8 threads on my Intel i3 14th gen. Example output:


## License

MIT

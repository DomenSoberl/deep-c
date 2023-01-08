# Deep-C

An implementation of deep learning algorithms in pure C programming language, without any third-party dependencies. All written from scratch. Suitable for embedded systems or low-level coding projects where TensorFlow and PyTorch cannot be used.

The library currently contains the following units:
- MLPC (Multilayer Perceptrons in C) - fully connected dense neural networks with SGD or Adam optimization.
- DDPGC (Deep Deterministic Policy Gradient in C) - deep Q-learning with continuous actions.

Examples:
- Learning the saddle function with MLPC.
- Swing up pendulum problem with DDPGC.

## Building and running on Linux

No prerequisites needed. Just run `make` in the top folder. The following files will be created:

- `./lib/mlpc.a` - the static MLPC library.
- `./lib/ddpgc.a` - the static DDPGC library.
- `./bin/saddle` - the saddle function executable.
- `./bin/pendulum` - the pendulum swing up executable.

## Building and running on Windows

Open the `./vs/deep-c.sln` solution in Visual Studio and build/run the desired example.

## Documentation

The provided `saddle` and `pendulum` examples should sufficiently demonstrate how to use the library. Additionally, `Doxygen` can be used to build the API documentation from the source code comments. Alternatively, you may examine the comments in the MLPC and DDPGC header files.

## Acnowledgements

If you find this code useful in your project/publication, please add an acknowledgements to this page.


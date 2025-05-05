# DynVision Documentation

## What is DynVision?

DynVision is a modular toolbox for building and evaluating recurrent convolutional neural networks (RCNNs) with biologically plausible temporal dynamics.
The toolbox provides researchers with a framework to explore how details in the network architecture influence temporal dynamics and shape visual processing, while handling most of the overhead to achieve computational efficiently.
The toolbox provides a collection of biologically-inspired components including:

- realistic lateral recurrent connections
- flexible skip and feedback connections
- activity evolution governed by dynamical systems equations
- unrolling of biological time with heterogenous time delays for different connection types 

## Documentation Structure

Our documentation is organized into four main categories:

1. **Tutorials**: Step-by-step guides for beginners to get started with DynVision
   - [Getting Started](getting-started.md): First steps with DynVision
   - [Basic Model Training](tutorials/basic-model-training.md): Train your first model
   - [Visualization Tutorial](tutorials/visualization-tutorial.md): Visualize model responses

2. **How-to Guides**: Task-oriented guides for solving specific problems
   - [Installation](user-guide/installation.md): Detailed installation instructions
   - [Custom Models](user-guide/custom-models.md): Define your own neural network architectures
   - [Data Processing](user-guide/data-processing.md): Work with different datasets
   - [Workflow Management](user-guide/workflows.md): Use Snakemake for experiments
   - [Model Evaluation](user-guide/evaluation.md): Evaluate model performance

3. **Reference**: Technical descriptions of DynVision's components
   - [Organization Overview](reference/organization.md): Structure of the toolbox
   - [Model Components API](reference/model-components.md): Core building blocks
   - [Recurrence Types](reference/recurrence-types.md): Different recurrent connection implementations
   - [Dynamics Solvers](reference/dynamics-solvers.md): ODE solvers for neural dynamics
   - [Configuration Reference](reference/configuration.md): Configuration file documentation

4. **Explanation**: Conceptual understanding of DynVision's approach
   - [Biological Plausibility](explanation/biological-plausibility.md): Alignment with neural systems
   - [Temporal Dynamics](explanation/temporal-dynamics.md): Understanding temporal properties
   - [Design Philosophy](explanation/design-philosophy.md): Core design principles

## Contributing

DynVision is an open-source project, and we welcome contributions! See our [Contributing Guide](contributing.md) for information on how to get involved.

## Getting Support

If you have questions or run into issues:

1. Check the [FAQ](user-guide/faq.md) for common questions
2. Search the [GitHub Issues](https://github.com/yourusername/dynvision/issues) to see if someone has encountered the same problem. Open a new issue if you can't find a solution.
3. Reach out via [Email](mailto:robin.gutzen@nyu.edu).

## License

DynVision is released under the MIT License. See the [LICENSE](https://github.com/yourusername/dynvision/blob/main/LICENSE) file for more details.

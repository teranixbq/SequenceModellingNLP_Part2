"""Data contoh untuk Task 1 text generation."""

TEXT_CORPUS = """
natural language processing studies how computers understand human language.
recurrent neural networks read a sentence one step at a time and keep a hidden
state for previous context. a simple rnn can learn short patterns, but long
sentences are difficult because gradients may vanish during training. lstm
models use input gates, forget gates, and output gates to decide what should be
remembered. this makes lstm useful for text generation, translation, and other
sequence modelling tasks. attention improves encoder decoder models by letting
the decoder focus on important source words while producing each target word.
in this lab we compare simple rnn and lstm language models, then build an
encoder decoder sequence to sequence model with bahdanau attention.
"""

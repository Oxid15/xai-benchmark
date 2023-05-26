Framework
=========

XAIB is build and aimed to be developed further using complete and defined framework of interpretability properties.
  
In the recent `work <https://arxiv.org/abs/2201.08164>`_ Nauta et al. proposes complete evaluation scheme for
XAI methods considering interpretability as a multi-faceted concept. They argue
that interpretability is not a binary property: it has different aspects that can be
measured independently. They propose "Co-12 properties" of interpretability for
XAI methods, which are namely: Correctness, Completeness, Consistency,
Continuity, Contrastivity, Covariate complexity, Compactness, Compositionality,
Confidence, Context, Coherence, and Controllability.
For each of them they suggest general ways of quantitatively measure them without 
concrete implementation suggestions and the overview of how different authors measure
them.
  
Authors define an ontology of interpretability which can serve as a framework for the development of
new quality metrics.
  
XAIB tries not to reinvent the wheel in the field of interpretability and uses this framework as
the most complete and recent to this day.

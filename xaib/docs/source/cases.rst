Cases
#####
:py:class:`xaib.base.Case`

Case is the core element of XAIB – it provides an interface to
conduct experiments.
  
It defines one abstract method which accepts name of an explainer, and
an explainer to evaluate. It receives a model and a dataset on creation. This is done
intentionally to insist on the particular usage workflow. User first defines all
settings – the model and data, which should be constant for one set of experiments
and then they can evaluate several explainer in the same settings using the same
model and data, which is correct way to compare different methods.
  
Another important aspect of Cases is that each one of them represents one abstract property
that is measured in it. These theoretically defined properties are complex and very
difficult to define numerically in exact way. This means that any metric will by
definition miss the property that we should measure. However, if several ways to
measure one property will be found (and they were) one property can and should
be represented using several exact metrics, which will save users from overfitting
even on the scale of one property.

Each case corresponds to one of the :doc:`Co-12 properties <framework>`.

Correctness
***********
:py:class:`xaib.cases.feature_importance.CorrectnessCase`

:py:class:`xaib.cases.example_selection.CorrectnessCase`

The property named Correctness means that explanations describe the model
correctly and truthfully. The explanations may not always be reasonable for the
user, but they should be true to the model to satisfy this criterion.

Metrics
=======
* :ref:`Feature importance: Parameter Randomization Check <metrics/feature_importance:Parameter randomization check>`
* :ref:`Example selection: Parameter Randomization Check <metrics/example_selection:Parameter randomization check>`


Completeness
************

Nothing in the explanation is left out, but it should be balanced with
compactness and correctness: "don't overwhelm". This property can be measured
from different points of view. For example reasoning completeness – white or
black box, can be measured only qualitatively and refers to the type of explanation.
Output-completeness – how the explanation covers output of model. Can be
measured using deletion check – delete or mask all important features and measure
the drop in model's performance.

A common definition of completeness revolves around showing the user as much relevant
information as possible, trying not to omit details of little influence in pursuit of a
pleasant explanation.

Consistency
***********

Identical inputs have identical explanations. This is different from
Continuity, because this metric is about implementation invariance. It can be from
example measured using implementation invariance check – construct identical
models that have different weights and check the closeness between explanations
made on them. This is based on the assumption that models with different values
of their internal state when trained on one task, will eventually have the same way
of reasoning that should be captured by explainer.

Continuity
**********
:py:class:`xaib.cases.feature_importance.ContinuityCase`

:py:class:`xaib.cases.example_selection.ContinuityCase`

How continuous explanation function is. Continuous functions are desirable,
because they are more predictable and comprehensive.

Metrics
=======
* :ref:`Feature importance: Small noise Check <metrics/feature_importance:Small noise check>`
* :ref:`Example selection: Small noise Check <metrics/example_selection:Small noise check>`


Contrastivity
*************
:py:class:`xaib.cases.feature_importance.ContrastivityCase`

:py:class:`xaib.cases.example_selection.ContrastivityCase`

How discriminative the explanation is in relation to different targets. The
contrast between different concepts is very important and explanation method
should explain instances of different classes in different ways.

Metrics
=======
* :ref:`Feature importance: Label difference <metrics/feature_importance:Label difference>`
* :ref:`Example selection: Target discriminativeness <metrics/example_selection:Target discriminativeness>`


Covariate complexity
********************
:py:class:`xaib.cases.feature_importance.CovariateComplexityCase`

:py:class:`xaib.cases.example_selection.CovariateComplexityCase`

The features used in the explanation should be comprehensible. Also non-
complex interactions between features are desired.

Metrics
=======
* :ref:`Feature importance: Covariate regularity <metrics/feature_importance:Covariate regularity>`
* :ref:`Example selection: Covariate regularity <metrics/example_selection:Covariate regularity>`


Compactness
***********
:py:class:`xaib.cases.feature_importance.CompactnessCase`

Compactness measures the size of explanations. Explanations should be
sparse, short and not redundant.

Compositionality
****************

Compositionality considers the format of presentation of the explanation.
Some formats are considered more interpretable than others. Here in some cases
metrics can be defined using for example Perceptual Realism – Fréchet Inception
Distance (FID) for example synthesis. The challenge is to define quantitative
measure for the presentation format of the explanations that will not involve
human experiments into computation.

Confidence
**********

Confidence describes the presence and accuracy of probability information.
It may be defined as some quality value for agreement between confidence and
true labels.

Context
*******

Context is about relevance for the users of different needs. May be computed
using Simulated User Studies – for example the identification of a better model or
bad features.

Coherence
*********
:py:class:`xaib.cases.feature_importance.CoherenceCase`

:py:class:`xaib.cases.example_selection.CoherenceCase`

To what extent the explanation is consistent with relevant background
knowledge, beliefs and general consensus. The agreement with domain-specific 
knowledge can be measured, but this is difficult to define and very task-dependent.

Metrics
=======
* :ref:`Feature importance: Other disagreement <metrics/feature_importance:Other disagreement>`
* :ref:`Example selection: Same class check <metrics/example_selection:Same class check>`

Controllability
***************

How interactive the explanation is for user. This can be measured using
Human Feedback Impact – improvement after feedback and the metric called
Concept-level feedback Satisfaction Ratio for example.

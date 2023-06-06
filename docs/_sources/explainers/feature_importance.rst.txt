Feature importance explainers
#############################

ConstantExplainer
*****************
:py:class:`xaib.explainers.feature_importance.constant_explainer.ConstantExplainer`

.. tags:: baseline

Returns the predefined constant vector for every input. This baseline is useful for
checking stability metrics. For example it should maximize :doc:`Model randomization check <../metrics/feature_importance>`

LimeExplainer
*************
:py:class:`xaib.explainers.feature_importance.lime_explainer.LimeExplainer`

Another popular feature importance approach is LIME. The concept is
that it tests what happens to the predictions of the model when it receives
variations of data. LIME generates a new dataset that consists of perturbed samples
and predictions of the black box model on them. On this new dataset LIME trains
an interpretable model, which is weighted by the proximity of the sampled
instances to the instance of interest. The learned model should be a good
approximation of the machine learning model predictions locally and explain
original model's predictions properly.

Source: `LIME <https://github.com/marcotcr/lime>`_

LinearRegressionExplainer
*************************
:py:class:`xaib.explainers.feature_importance.linear_regression_explainer.LinearRegressionExplainer`

.. tags:: white_box

This explainer gives results by the multiplying the input by coefficients of a trained linear regression.

RandomExplainer
***************
:py:class:`xaib.explainers.feature_importance.random_explainer.RandomExplainer`

.. tags:: baseline

Good baseline to represent average results of an explainer. Output distribution can be
shifted and scaled to test for normalization.

ShapExplainer
*************
:py:class:`xaib.explainers.feature_importance.shap_explainer.ShapExplainer`

The shap method is considered as a method that can be applied to explain
individual predictions of any ML model. Shap is based on a game theory approach
to additively accumulate the contribution of all features involved in a model.
This method assigns each feature an “importance value” within a set of conditional
expectations for a particular prediction. The results of this additive procedure are
called “shap values”. These values can be spread across from a “base value”
(which represents the average of the observations). As one can see, since the shap
method accounts for all features and randomness of their order, this method can be
quite computationally expensive for large models.

Source `shap <https://github.com/slundberg/shap>`_

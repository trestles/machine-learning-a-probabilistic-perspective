# machine-learning-a-probabilistic-perspective

### 1. Introduction
- 1.1. Machine learning: what and why? 1
  - 1.1.1. Types of machine learning 2
- 1.2. Supervised learning 3
  - 1.2.1. Classification 3
  - 1.2.2. Regressin 8
- 1.3. Unsupervised learning 9
  - 1.3.1. Discovering clusters 10
  - 1.3.2. Discovering latent factors 11
  - 1.3.3. Discovering graph structure 13
  - 1.3.4. Matrix completion 14
- 1.4. Some basic concepts in machine learning 16
  - 1.4.1. Parametric vs non-parametric models 16
  - 1.4.2. A simple non-parametric classifier: K-nearest neighbors
  - 1.4.3. The curse of dimensionality 18
  - 1.4.4. Parametric models for classification and regression 19
  - 1.4.5. Linear regression 19
  - 1.4.6. Logistic regression 21
  - 1.4.7. Overfitting 22
  - 1.4.8. Model selection 22
  - 1.4.9. No free lunch theorem 24

### 2. Probability
- 2.1. Introduction 27
- 2.2. A brief review of probability theory 28
  - 2.2.1. Discrete random variables 28
  - 2.2.2. Fundamental rules 29
  - 2.2.3. Bayes' rule 29
  - 2.2.4. Independence and conditional independence 31
  - 2.2.5. Continuous random variables 32
  - 2.2.6. Quantiles 33
  - 2.2.7. Mean and variance 33
- 2.3. Some common discrete distributions 34
  - 2.3.1. The binomial and Bernoulli distributions 34
  - 2.3.2. The multinomial and multinoulli distributions 35
  - 2.3.3. The Poisson distribution 37
  - 2.3.4. The emperical distribution 37
- 2.4. Some common continuous distributions 38
  - 2.4.1. Gaussian (normal) distribution 38
  - 2.4.2. Degenerate pdf 39
  - 2.4.3. The Student's t distribution 39 
  - 2.4.4. The Laplace distribution 41
  - 2.4.5. The gamma distribution 41
  - 2.4.6. The beta distribution 43
  - 2.4.7. Pareto distribution 43
- 2.5. Joint probability distributions 44
  - 2.5.1. Covariance and correlation 45
  - 2.5.2. The multivariate Gaussian 46
  - 2.5.3. Multivariate Student t distribution 47
  - 2.5.4. Dirichlet distribution 49
- 2.6. Transformations of random variables 49
  - 2.6.1. Linear transformations 49
  - 2.6.2. General transformations 50
  - 2.6.3. Central Limit Theorem 52
- 2.7. Monte Carlo approximation 53
  - 2.7.1. Example: change of variables, the MC way 53
  - 2.7.2. Example: estimating pi by Monte Carlo integration 54
  - 2.7.3. Accuracy of Monte Carlo approximation 54
- 2.8. Information theory
  - 2.8.1. Entropy 
  - 2.8.2. KL Divergence 58
  - 2.8.3. Mutual information 59

### 3. Generative Models for Discrete Data
- 3.1. Introduction 67
- 3.2. Bayesian concept learning 67
  - 3.2.1. Likelihood 69
  - 3.2.2. Prior 69
  - 3.2.3. Posterior 70
  - 3.2.4. Posterior predictive distribution 73
  - 3.2.5. A more complex prior 74
- 3.3. The beta-binomial model 74
  - 3.3.1. Likelihood 75
  - 3.3.2. Prior 76
  - 3.3.3. Posterior 77
  - 3.3.4. Posterior predictive distribution 79
- 3.4. The Dirichlet-multinomial model 80
  - 3.4.1. Likelihood 81
  - 3.4.2. Prior 81
  - 3.4.3. Posterior 81
  - 3.4.4. Posterior predictive
- 3.5. Naive Bayes classifiers 84
  - 3.5.1. Model fitting 85
  - 3.5.2. Using the model for prediction 87
  - 3.5.3. the log-sum-exp trick 88
  - 3.5.4. Feature selection using mutual information 89
  - 3.5.5. Classifying documents using bag of words 90 

### 4. Gaussian Models
- 4.1. Introduction 99
  - 4.1.1. Notation 99
  - 4.1.2. Basics 99
  - 4.1.3. MLE for an MVN 101
  - 4.1.4. Maximum entropy deviation of the Gaussian * 103
- 4.2. Gaussian discriminant analysis 103
  - 4.2.1. Quadratic discriminant analysis (QDA) 104
  - 4.2.2. Linear discriminant analysis (LDA) 105
  - 4.2.3. Two-class LDA 106
  - 4.2.4. MLE for discriminant analysis 108
  - 4.2.5. Strategies for preventing overfitting 108
  - 4.2.6. Regularized LDA * 109
  - 4.2.7. Diagonal LDA 110
  - 4.2.8. Nearest shrunken centroids classifier * 111
- 4.3. Inferences in jointly Gaussian distributions 112
  - 4.3.1. Statement of the result 113
  - 4.3.2. Examples 113
  - 4.3.3. Information form 117
  - 4.3.4. Proof of the result * 118
- 4.4. Linear Gaussian systems  121
  - 4.4.1. Statement of the result 122
  - 4.4.2. Examples 122
  - 4.4.3. Proof of the result * 127
- 4.5. Digression: The Wishart distribution * 128
  - 4.5.1. Inverse Wishart distribution 129
  - 4.5.2. Visualizing the Wishart distribution 129
- 4.6. Inferring the parameters of an MVN 129
  - 4.6.1. Posterior distribution of mu 130 
  - 4.6.2. Posterior distribution of E * 131
  - 4.6.3. Posterior distribution of u nad E * 134
  - 4.6.4. Sensor fusion with unknown precisions * 140


### 5. Bayesian Statistics
- 5.1. Introduction
- 5.2. Summarizing posterior distributions 151
  - 5.2.1. MAP estimation 151
  - 5.2.2. Credible intervals 154
  - 5.2.3. Inference for a difference in proportions 156
- 5.3. Bayesian model selection 157
  - 5.3.1. Bayesian Occam's razor 158
  - 5.3.2. Computing the marginal likelihood (evidence) 160
  - 5.3.3. Bayes factors 165
  - 5.3.4. Jeffrey's-Lindley paradox * 166
- 5.4. Priors 167
  - 5.4.1. Uninformative priors 167
  - 5.4.2. Jeffreys priors * 168
  - 5.4.3. Robust priors 170 
  - 5.4.4. Mixture of conjugate priors 171
- 5.5. Hieararchical Bayes 173
  - 5.5.1. Example: modeling related cancer rates 173
- 5.6. Emperical Bayes 174
  - 5.6.1. Example: beta-binomial model 175
  - 5.6.2. Example: Gaussian-Gaussian model 176
- 5.7. Bayesian Decision Theory 178
  - 5.7.1. Bayes estimators for common loss functions 179
  - 5.7.2. The false positive vs false negative tradeoff 182
  - 5.7.3. Other topics * 186


### 6. Frequentist Statistics
- 6.1. Introduction
- 6.2. Sampling distribution of an estimator 
  - 6.2.1 Bootstrap 194
  - 6.2.2. Large sample theory for the MLE * 195
- 6.3. Frequentist decision theory 197
  - 6.3.1. Bayes risk 197
  - 6.3.2. Minimax risk 198
  - 6.3.3. Admissable estimators 199
- 6.4. Desirable properties of estimators 202
  - 6.4.1. Consistent estimators 202
  - 6.4.2. Unbiased estimators 203
  - 6.4.3. Maximum variance estimators 203
  - 6.4.4. The bias-variance tradeoff 204
- 6.5. Emperical risk minimization 207
  - 6.5.1. Regularized risk minimization 208
  - 6.5.2. Structural risk minimization 208
  - 6.5.3. Estimating the risk using cross-validation 209
  - 6.5.4. Upper-bounding the risk using statistical learning theory * 211
  - 6.5.5. Surrogate loss functions 213
- 6.6. Pathologies of frequentist statistics * 214 
  - 6.6.1. Counter-intuitive behavior of confidence intervals 214
  - 6.6.2. p-values considered harmful 215
  - 6.6.3. The likelihood principle 217
  - 6.6.4. Why isn't everyone a Bayesian? 217

### 7. Linear Regression
- 7.1. Introduction 219
- 7.2. Model Specification 219
- 7.3. Maximum likelihood estimation (least squares) 219
  - 7.3.1. Derivation of the MLE 221
  - 7.3.2. Geometric interpretation 222
  - 7.3.3. Convexity 223
- 7.4. Robust linear regression * 225
- 7.5. Ridge Regression 227
  - 7.5.1. Basic idea 227
  - 7.5.2. Numerically stable computation * 229
  - 7.5.3. Connection with PCA * 230
  - 7.5.4. Regularization effects of big data 232
- 7.6. Bayesian linear regression 233
  - 7.6.1. Computing the posterior 234
  - 7.6.2. Computing the posterior predictive 235
  - 7.6.3. Bayesian inference when  is unknown 236
  - 7.6.4. EB for linear regression (evidence procedure) 240

### 8. Logistic Regression
- 8.1. Introduction 247
- 8.2. Model specification 247
- 8.3. Model fitting 248
  - 8.3.1. MLE 249
  - 8.3.2. Steepest descent 249
  - 8.3.3. Newton's method 251
  - 8.3.4. Iteratively reweighted least squares (IRLS) 253
  - 8.3.5. Quasi-Newton (variable metric) methods 253
  - 8.3.6. l2 regularization 254
  - 8.3.7. multi-class logistic regression 255
-8.4. Bayesian logistic regression 257
  - 8.4.1. Laplace approximation 257
  - 8.4.2. Deviation of the Bayesian information criterion (BIC) 258
  - 8.4.3. Gaussian approximation for logistic regression 258
  - 8.4.4. Approximating the posterior predictive 260
  - 8.4.5. Residual analysis (outlier detection) * 263
- 8.5. Online learning and stochastic optimization 264
  - 8.5.1. Online learning and regret minimization 264
  - 8.5.2. Stochastic optimization and risk minimization 265
  - 8.5.3. The LMS algorithm 267
  - 8.5.4. The perceptron algorithm 268
  - 8.5.5. A Bayesian view 270
- 8.6. Generative vs Discriminative classifiers 270
  - 8.6.1. Pros and cons of each approach 271
  - 8.6.2. Dealing with missing data 271
  - 8.6.3. Fisher's linear discriminant analysis (FLDA) * 274


### 9. Generalized Linear Models and the Exponential Family
- 9.1. Introduction 283
- 9.2. The exponential family 283
  - 9.2.1. Definition 284
  - 9.2.2. Examples 284
  - 9.2.3. Log partition function 286
  - 9.2.4. MLE for the exponential function 288
  - 9.2.5. Bayes for the exponential family * 289
  - 9.2.6. Maximum entropy derivation of the exponential family * 291
- 9.3. Generalized linear models (GLM) 292
  - 9.3.1. Basics 292
  - 9.3.2. ML and MAP estimation 294
  - 9.3.3. Bayesian inference 295
- 9.4. Probit regression 295
  - 9.4.1. ML/MAP estimation using gradient-based optimization 296
  - 9.4.2. Latent variable interpretation 296
  - 9.4.3. Oridinal probit regression * 297
  - 9.4.4. Multinomial probit models * 297
- 9.5. Mult-task learning 298
  - 9.5.1. Hierarchical Bayes for multi-task learning 298
  - 9.5.2. Application to personalized email spam filtering 298
  - 9.5.3. Application to domain adaptation 299
  - 9.5.4. Other kinds of prior 299
- 9.6. Generalized linear mixed models * 300
  - 9.6.1. Example: semi-parametric GLMMs for medical data 300
  - 9.6.2. Computational issues 302
- 9.7. Learning to rank * 302
  - 9.7.1. The pointwise approach 303
  - 9.7.2. The pairwise approach 303
  - 9.7.3. The listwise approach 304 
  - 9.7.4. Loss functions for ranking 305

### 10. Directed Graphical Models (Bayes nets)
- 10.1 Introduction 309
  - 10.1.1. Chain rule 309
  - 10.1.2. Conditional independence 310
  - 10.1.3. Graphical models 310
  - 10.1.4. Graph terminology 311
  - 10.1.5. Directed graphical models 312
- 10.2. Examples 313
  - 10.2.1. Naive Bayes classifiers 313
  - 10.2.2. Markov and hidden Markov models 314
  - 10.2.3. Medical diagnosis 315
  - 10.2.4. Genetic linkage analysis * 317
  - 10.2.5. Directed Gaussian graphical models * 320 
- 10.3. Inference 321
- 10.4. Learning 322
  - 10.4.1. Plate notation 322
  - 10.4.2. Learning from complete data 324
  - 10.4.3. Learning with missing and / or latent variables 325 
- 10.5. Conditional independence properties of DGMs 326
  - 10.5.1. d-separation and the Bayes Ball algorithm (global Markov properties) 326
  - 10.5.2. Other Markov properties of DGMs 329
  - 10.5.3. Markov blanket and full conditionals 329
- 10.6. Influence of decision diagrams * 330


### 25. Clustering 877
- 25.1 Introduction 877
  - 25.1.1 Measuring (dis)similarity 877
  - 25.1.2 Evaluating the output of clustering methods 878
- 25.2 Dirichlet process mixture models 881
  - 25.2.1. From finite to infinite mixture models 881
  - 25.2.2. The Dirichlet process 884
  - 25.2.3. Applying Dirichlet processes to mixture modeling 887
  - 25.2.4. Fitting a DP mixture model 888
- 25.3. Affinity propogation
- 25.4 Spectral clustering 892
  - 25.4.1 Graph Laplacian 893
  - 25.4.2. Normalized graph Laplacian 894
  - 25.4.3. Example 895
- 25.5. Hierarchical Clustering 895
  - 25.5.1. Agglomerative clustering 897
  - 25.5.2. Divisive clustering 900
  - 25.5.3. Choosing the number of clusters 901
  - 25.5.4. Bayesian hierarchical clustering 901
- 25.6. Clustering datapoints and features 903
  - 25.6.1. Biclustering 905
  - 25.6.2. Multi-view clustering 905

### 26 Graphical model structure learning 909
- 26.1. Introduction 909
- 26.2. Structure learning for knowledge discovery 910
  - 26.2.1. Relevance networks 910
  - 26.2.2. Dependency networks 911
- 26.3. Learning tree structures 912
  - Directed or undirected tree? 913
  - Chow-Liu algorithm for finding the ML tree structure 914
  - Finding the MAP forest 914
  - Mixture of trees 916
- 26.4. Learning DAG structures 916
  - 26.4.1. Markov equivalence 916
  - 26.4.2. Exact structural inference 918
  - 26.4.3. Scaling up to larger graphs 922
- 26.5. Learning DAG structure with latent variables 924
  - 26.5.1. Approximating the marginal likelihood when we have missing data 924
  - 26.5.2. Structural EM 927
  - 26.5.3. Discovering hidden variables 928
  - 26.5.4. Case study: Google's Rephil 930
  - 26.5.5. Structural equation models * 931
- 26.6 Learning causal DAGs 933
  - 26.6.1. Causal interpretation of DAGs 933
  - 26.6.2. Using causal DAGs to resolve Simpson's paradox 935
  - 26.6.3. Learning causal DAG structures 938
- 26.7. Learning undirected Gaussian graphical models 940
  - 26.7.1. MLE for a GGM 940
  - 26.7.2. Graphical lasso 941
  - 26.7.3. Bayesian inference for GGM structure * 943
  - 26.7.4. Handling non-Gaussian data using copulas * 944
- 26.8. Learning undirected discrete graphical models 944
  - 26.8.1. Graphical lasso for MRFs / CRFs 944
  - 26.8.2. Thin junction trees 945

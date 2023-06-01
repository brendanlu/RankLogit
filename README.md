# RankLogit
When performing 'Latent Class Analysis' segmentation using QSoftware, Q will automatically fit a 'Tied Ranking Logistic Model' to 'Rank' question types. Whilst some data might not be naturally or intuitively thought of as 'ranked', ranking transformations often uncover fascinating and valuable results and insights. Maximum Likelihood Estimation of this model can be accomplised with partial likelihood procedures, which are available in SAS, BMDP, SYSTAT and SPSS (Allison and Christakis, 1994). 

Once parameterised, there are also some commercial programs that implement the likelihood. 1) The PHREG procedure in SAS, and 2) QSoftware (in the backend to compute posterior latent class probabilities). However, following segmentation analysis, one may wish to use the fitted model for further purposes such as labelling new  website traffic in real time, or to fit various models and compare the segment labelling outcomes in a CSV file or Jupyter notebook. In these cases, the commercial implementations are no good (not lightweight and cannot easily integrate into website back end; cost subscription money; typically have to work with more complex data file types specific to market research - not so easy to manipulate).

This repo contains an old bit of Python code which is a bit clunky, but contains a correct, and relatively quick implementation of the rather involved likelihood function. The Python module contains some clunky OOP code, and implementation of simple Bayesian logic, to use the computed likelihoods to estimate posterior latent class membership. 

**Reference:** 
Allison, P.D. and Christakis, N.A. (1994). Logit Models for Sets of Ranked Items. Sociological Methodology, 24, p.199. doi:https://doi.org/10.2307/270983.

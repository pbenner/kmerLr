## Installation

Pre-compiled binaries are available [here](https://github.com/pbenner/kmerLr-binary).

## Documentation

Benner, Philipp. *Computing Leapfrog Regularization Paths with Applications to Large-Scale K-mer Logistic Regression.* Journal of Computational Biology (2021).

## KmerLr Example

Foreground and background data must be stored in seperate files where the rows are the samples and the columns the features:
```bash
$ cat test_fg.table 
0.4,1.4,10.5,0.21, 5.1,0.22,4.21
0.2,1.2,33.5,0.23, 1.6,0.11,5.23
0.3,1.3,20.3,0.26, 4.4,0.13,4.25
0.4,1.2,36.5,0.29,10.1,0.11,2.21
0.2,1.2,65.1,0.22,50.1,0.31,4.23
0.1,1.3,70.2,0.21, 0.2,0.21,2.21
0.3,1.2,43.5,0.22, 0.1,0.22,4.22
0.2,1.4,10.3,0.23, 0.4,0.31,2.32
$ cat test_bg.table 
0.2,22.4,30.5,0.53,13.3,0.11,14.21
0.4,11.5,13.1,0.54, 3.6,0.42,45.23
0.5,15.4,24.2,0.44, 1.6,0.31,24.25
0.2,21.3,26.5,0.20, 0.3,0.14,21.21
0.4,11.1,10.3,0.20,10.4,0.61,14.23
0.2,31.4,50.2,0.11,65.3,0.18,12.21
1.1,11.2,32.5,0.13, 2.1,0.32,14.22
1.3,13.4,70.3,0.45, 6.4,0.99,12.32
```

Estimate a logistic regression model with two features:
```bash
$ ./kmerLr -v --type=scoresLr learn --lambda-auto=2 --save-trace test_{fg,bg}.table test
Reading scores from `test_fg.table'... done
Reading scores from `test_bg.table'... done
Estimating classifier with 2 non-zero coefficients...
Estimating parameters with lambda=2.496875e+00...
Estimated classifier has 2 non-zero coefficients, selecting 0 new features...
Exporting trace to `test.trace'... done
Exporting model to `test_2.json'... done
```

Print the coefficients of the estimates model:
```bash
$ ./kmerLr -v --type=scoresLr coefficients test_2.json
Importing distribution from `test.json'... done
     1  -5.466291e-02 2
     2  -3.026280e-02 7
```

## Regularization Paths

Estimation of regularization paths:
```bash
$ ./kmerLr -v --type=scoresLr learn --lambda-auto=2,5 --save-path test_{fg,bg}.table test
Reading scores from `test_fg.table'... done
Reading scores from `test_bg.table'... done
Estimating classifier with 2 non-zero coefficients...
Estimating parameters with lambda=2.496875e+00...
Estimated classifier has 2 non-zero coefficients, selecting 0 new features...
Estimating classifier with 5 non-zero coefficients...
Estimating parameters with lambda=4.870316e-02...
Estimated classifier has 3 non-zero coefficients, selecting 2 new features...
Estimating parameters with lambda=1.891759e-03...
Estimated classifier has 3 non-zero coefficients, selecting 2 new features...
Estimating parameters with lambda=1.482768e-04...
Estimated classifier has 5 non-zero coefficients, selecting 0 new features...
Exporting regularization path to `test.path'... done
Exporting model to `test_2.json'... done
Exporting model to `test_5.json'... done
```
Plot regularization path:
```R
library(RColorBrewer)

plot.path <- function(filename, col=brewer.pal(n = 8, name = "RdBu")) {
    t <- read.table(filename, header=TRUE)
    x <- t$norm
    y <- read.csv(textConnection(as.character(t$theta)), header=FALSE)
    matplot(x, y, type="l", lty=1, lwd=1.5, col=col, xlab=expression(paste("||", theta, "||")[1]), ylab=expression(theta[i]), xlim=c(min(x), max(x)))
    abline(v=x, lty=2, col="lightgray")
}

plot.path("test.path")
```

## Cross-Validation

Use a logistic regression model with two features in a 5-fold cross-validation:
```bash
$ ./kmerLr -v --type=scoresLr learn --epsilon-loss=1e-4 --lambda-auto=2,5 --save-trace --save-path --k-fold-cv=5 test_{fg,bg}.table test
Reading scores from `test_fg.table'... done
Reading scores from `test_bg.table'... done
Estimating classifier with 2 non-zero coefficients...
Estimating parameters with lambda=3.185208e+00...
Estimated classifier has 1 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=2.976290e+00...
Estimated classifier has 2 non-zero coefficients, selecting 0 new features...
Estimating classifier with 5 non-zero coefficients...
Estimating parameters with lambda=6.695171e-02...
Estimated classifier has 4 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=1.331785e-02...
Estimated classifier has 4 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=1.012429e-02...
Estimated classifier has 4 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=1.978671e-02...
Estimated classifier has 5 non-zero coefficients, selecting 0 new features...
Exporting trace to `test_0.trace'... done
Exporting regularization path to `test_0.path'... done
Exporting model to `test_2_0.json'... done
Exporting model to `test_5_0.json'... done
Estimating classifier with 2 non-zero coefficients...
Estimating parameters with lambda=3.678846e+00...
Estimated classifier has 2 non-zero coefficients, selecting 0 new features...
Estimating classifier with 5 non-zero coefficients...
Estimating parameters with lambda=3.088860e-02...
Estimated classifier has 4 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=9.727014e-03...
Estimated classifier has 5 non-zero coefficients, selecting 0 new features...
Exporting trace to `test_1.trace'... done
Exporting regularization path to `test_1.path'... done
Exporting model to `test_2_1.json'... done
Exporting model to `test_5_1.json'... done
Estimating classifier with 2 non-zero coefficients...
Estimating parameters with lambda=3.032692e+00...
Estimated classifier has 2 non-zero coefficients, selecting 0 new features...
Estimating classifier with 5 non-zero coefficients...
Estimating parameters with lambda=1.763804e-02...
Estimated classifier has 4 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=6.953048e-03...
Estimated classifier has 5 non-zero coefficients, selecting 0 new features...
Exporting trace to `test_2.trace'... done
Exporting regularization path to `test_2.path'... done
Exporting model to `test_2_2.json'... done
Exporting model to `test_5_2.json'... done
Estimating classifier with 2 non-zero coefficients...
Estimating parameters with lambda=2.848077e+00...
Estimated classifier has 2 non-zero coefficients, selecting 0 new features...
Estimating classifier with 5 non-zero coefficients...
Estimating parameters with lambda=3.217164e-02...
Estimated classifier has 4 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=6.634468e-03...
Estimated classifier has 5 non-zero coefficients, selecting 0 new features...
Exporting trace to `test_3.trace'... done
Exporting regularization path to `test_3.path'... done
Exporting model to `test_2_3.json'... done
Exporting model to `test_5_3.json'... done
Estimating classifier with 2 non-zero coefficients...
Estimating parameters with lambda=2.543269e+00...
Estimated classifier has 1 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=2.581854e+00...
Estimated classifier has 2 non-zero coefficients, selecting 0 new features...
Estimating classifier with 5 non-zero coefficients...
Estimating parameters with lambda=4.322065e-02...
Estimated classifier has 4 non-zero coefficients, selecting 1 new features...
Estimating parameters with lambda=7.162611e-03...
Estimated classifier has 5 non-zero coefficients, selecting 0 new features...
Exporting trace to `test_4.trace'... done
Exporting regularization path to `test_4.path'... done
Exporting model to `test_2_4.json'... done
Exporting model to `test_5_4.json'... done
Exporting cross-validation results to `test_2.table'... done
Exporting cross-validation results to `test_5.table'... done
```

Plot cross-validation result:
```R
library(ROCR)

plot.roc <- function(filename) {
    t <- read.table(filename, header=T)
    p <- prediction(t$prediction, t$labels)
    p <- performance(p, "tpr", "fpr")
    plot(p)
}

plot.roc("test_2.table")
```

## Objective function

<a href="https://www.codecogs.com/eqnedit.php?latex=\omega(\theta)&space;=&space;-\frac{1}{n}\sum_{i=1}^n&space;\left\{y_i&space;\log\sigma(x_i\theta)&space;&plus;&space;(1-y_i)\log(1-\sigma(x_i\theta))\right\}&space;&plus;&space;\lambda&space;\left&space;\Vert&space;\theta&space;\right\Vert_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\omega(\theta)&space;=&space;-\frac{1}{n}\sum_{i=1}^n&space;\left\{y_i&space;\log\sigma(x_i\theta)&space;&plus;&space;(1-y_i)\log(1-\sigma(x_i\theta))\right\}&space;&plus;&space;\lambda&space;\left&space;\Vert&space;\theta&space;\right\Vert_1" title="\omega(\theta) = -\frac{1}{n}\sum_{i=1}^n \left\{y_i \log\sigma(x_i\theta) + (1-y_i)\log(1-\sigma(x_i\theta))\right\} + \lambda \left \Vert \theta \right\Vert_1" /></a>

## Symbolic regression

Primary input features:
```bash
$ cat scoresLr_test_primary_fg.table
a,b,c,d
0.4,1.4,10.5,0.21
0.2,1.2,33.5,0.23
0.3,1.3,20.3,0.26
0.4,1.2,36.5,0.29
0.2,1.2,65.1,0.22
0.1,1.3,70.2,0.21
0.3,1.2,43.5,0.22
0.2,1.4,10.3,0.23
$ cat scoresLr_test_primary_bg.table
a,b,c,d
0.2,22.4,30.5,0.53
0.4,11.5,13.1,0.54
0.5,15.4,24.2,0.44
0.2,21.3,26.5,0.20
0.4,11.1,10.3,0.20
0.2,31.4,50.2,0.11
1.1,11.2,32.5,0.13
1.3,13.4,70.3,0.45
```

Use *kmerLr* to expand feature set:
```bash
kmerLr --type=scoresLr expand --max-features=20 --header scoresLr_test_primary_fg.table,scoresLr_test_primary_bg.table scoresLr_test_expanded
```

Result:
```bash
$ cat scoresLr_test_expanded_0.table
a,b,c,d,exp(a),exp(b),exp(c),exp(d),exp(-a),exp(-b),exp(-c),exp(-d),log(a),log(b),log(c),log(d),(a^2),(b^2),(c^2),(d^2)
4.000000e-01,1.400000e+00,1.050000e+01,2.100000e-01,1.491825e+00,4.055200e+00,3.631550e+04,1.233678e+00,6.703200e-01,2.465970e-01,2.753645e-05,8.105842e-01,-9.162907e-01,3.364722e-01,2.351375e+00,-1.560648e+00,1.600000e-01,1.960000e+00,1.102500e+02,4.410000e-02
2.000000e-01,1.200000e+00,3.350000e+01,2.300000e-01,1.221403e+00,3.320117e+00,3.538874e+14,1.258600e+00,8.187308e-01,3.011942e-01,2.825757e-15,7.945336e-01,-1.609438e+00,1.823216e-01,3.511545e+00,-1.469676e+00,4.000000e-02,1.440000e+00,1.122250e+03,5.290000e-02
3.000000e-01,1.300000e+00,2.030000e+01,2.600000e-01,1.349859e+00,3.669297e+00,6.549045e+08,1.296930e+00,7.408182e-01,2.725318e-01,1.526940e-09,7.710516e-01,-1.203973e+00,2.623643e-01,3.010621e+00,-1.347074e+00,9.000000e-02,1.690000e+00,4.120900e+02,6.760000e-02
4.000000e-01,1.200000e+00,3.650000e+01,2.900000e-01,1.491825e+00,3.320117e+00,7.108019e+15,1.336427e+00,6.703200e-01,3.011942e-01,1.406862e-16,7.482636e-01,-9.162907e-01,1.823216e-01,3.597312e+00,-1.237874e+00,1.600000e-01,1.440000e+00,1.332250e+03,8.410000e-02
2.000000e-01,1.200000e+00,6.510000e+01,2.200000e-01,1.221403e+00,3.320117e+00,1.873142e+28,1.246077e+00,8.187308e-01,3.011942e-01,5.338623e-29,8.025188e-01,-1.609438e+00,1.823216e-01,4.175925e+00,-1.514128e+00,4.000000e-02,1.440000e+00,4.238010e+03,4.840000e-02
1.000000e-01,1.300000e+00,7.020000e+01,2.100000e-01,1.105171e+00,3.669297e+00,3.072364e+30,1.233678e+00,9.048374e-01,2.725318e-01,3.254823e-31,8.105842e-01,-2.302585e+00,2.623643e-01,4.251348e+00,-1.560648e+00,1.000000e-02,1.690000e+00,4.928040e+03,4.410000e-02
3.000000e-01,1.200000e+00,4.350000e+01,2.200000e-01,1.349859e+00,3.320117e+00,7.794889e+18,1.246077e+00,7.408182e-01,3.011942e-01,1.282892e-19,8.025188e-01,-1.203973e+00,1.823216e-01,3.772761e+00,-1.514128e+00,9.000000e-02,1.440000e+00,1.892250e+03,4.840000e-02
2.000000e-01,1.400000e+00,1.030000e+01,2.300000e-01,1.221403e+00,4.055200e+00,2.973262e+04,1.258600e+00,8.187308e-01,2.465970e-01,3.363310e-05,7.945336e-01,-1.609438e+00,3.364722e-01,2.332144e+00,-1.469676e+00,4.000000e-02,1.960000e+00,1.060900e+02,5.290000e-02
```

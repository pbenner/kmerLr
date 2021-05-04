## Installation

Pre-compiled binaries are available [here](https://github.com/pbenner/kmerLr-binary).

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

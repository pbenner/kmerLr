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


$ ./kmerLr -v --type=scoresLr learn --lambda-auto=2 --save-trace test_{fg,bg}.table test
Reading scores from `test_fg.table'... done
Reading scores from `test_bg.table'... done
Fitting data transform... done
Selecting 2 features... done
Normalizing data... done
Estimating parameters with lambda=4.647556...
Estimated classifier has 2 non-zero coefficients, selecting 0 new features... done
Exporting model to `test.json'... done

$ ./kmerLr -v --type=scoresLr coefficients test.json 
Importing distribution from `test.json'... done
     1  -5.786522e-01 2 
     2  -1.642988e-02 7 

$ ./kmerLr -v --type=scoresLr learn --lambda-auto=2 --save-trace --k-fold-cv=5 test_{fg,bg}.table test

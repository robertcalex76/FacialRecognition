README:

1) You will be prompted to run test folder or validation.
        You will want to run test folder for almost all test runs. Validation is used to make sure that the classifiers are working correctly.

2) Then you will be prompted to enter which data set you would like to run your test on.
        1 run of Friends_Family in MLP can take 1 minute, big_test can take 8 minutes, all can take 30 minutes

3) Then you will be prompted to select the classifier to use

4) You will then be prompte to turn on shuffle or not
        Shuffle is used to test the generalization performance of the program. Details are described in the report

        4a) Shuffle on: enter the number of test rounds
                This number is used to average out performance data over a range of tests
                This number nearly multiplies runtimes by whatever number you enter

Results are printed at the end of the program.

File System:

    Add training images you would like to enter in the Friends_Family folder, big_test, or all.
    Then add testing images into TEST_IMAGES folder. 
    For correct formatting and performance, training images are saved in format: first_last_xxxx.jpg (xxxx being count ie 0000, 0001) (jpg or png or jpeg)
    Testing images should be saved under the format: first_last_0 (please only include 1 testing image per class (person))
    The program should handle the calculations and predictions from this point

    Test_output will store all saved heads from the TEST_IMAGES folder
    heads is a temporary folder that saves all heads from training data in the previous run. Auto cleaning this folder can be toggled in the code. 

Any additional information can be found in the report. 

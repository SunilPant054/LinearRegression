from athelete import Athelete


class PlayerMain:
    
    X1, X2, Y = Athelete.dataSplit()
    Athelete.preDataVisualization(X1, X2, Y)
    X_train, X_test, y_train, y_test = Athelete.dataSplitTrainTest()
    Athelete.trainModel(X_train, X_test, y_train, y_test)
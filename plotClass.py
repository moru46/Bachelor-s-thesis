




class plotClass:

    def __init__(self, var1, var2, var3):
        self.data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.trained_model = None
        self.var_test_size = var1
        self.var_random_state = var2
        self.var_n_estimators = var3

        # carico il file csv contenente il mio Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing


class RandomForestClass:

    # i valori di test per test_size,random_state e n_estimators sono passati da main
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

    def caricaCSV(self):
        self.data = pd.read_csv(r'C:\Users\utente\Desktop\UNIPI\Tesi\istanbulCalls.csv')
        

    def classDivision1(self, tipoClasse):
        if tipoClasse == 1:
            # converto i CASESTUDY in int
            self.data.replace(to_replace=['UNIVERSITY', 'LEVENT', 'MASLAK - INDUSTRIES', 'TAKSIM - SHOPPING',
                                          'HISTORIC - TOURISM'], value=[1, 2, 3, 4, 5], inplace=True)
            self.x = self.data.values
            # self.data.replace(to_replace=['MASLAK - INDUSTRIES', 'TAKSIM - SHOPPING', 'HISTORIC - TOURISM'],
            #                 value=['MASLAK', 'TAKSIM', 'HISTORIC'], inplace=True)
            # self.y contiene gli hotspot
            self.y = self.data['CASESTUDY'].values
            # elimino la colonna CASESTUDY da self.x, e la colonna relativa ai cluster(feriale,festivo) perchè in questo
            # momento non la reputo utile per questo tipo di sperimentazione
            # aggiornato il codice rimuovendo anche i dati relativi alle date
            self.x = np.delete(self.x, 0, axis=1)  # casestudy
            self.x = np.delete(self.x, 0, axis=1)  # mese
            self.x = np.delete(self.x, 0, axis=1)  # giorno
            self.x = np.delete(self.x, 0, axis=1)  # WE/WD
            # print(self.x)
        elif tipoClasse == 2:
            self.data.replace(to_replace=['MASLAK - INDUSTRIES', 'TAKSIM - SHOPPING', 'HISTORIC - TOURISM'],
                              value=['MASLAK', 'TAKSIM', 'HISTORIC'], inplace=True)
            # valuto la possibilità di introdurre 10 classi da catalogare, considerando in questo caso anche i relativi
            # cluster ovvero giorni feriali e festivi; 0 se il giorno è feriale(lavorativo) e 1 se festivo
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'UNIVERSITY') & (self.data['WEEKEND'] == 0),
                                              'UNIVERSITY_WD', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'UNIVERSITY') & (self.data['WEEKEND'] == 1),
                                              'UNIVERSITY_WE', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'MASLAK') & (self.data['WEEKEND'] == 0),
                                              'MASLAK_WD', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'MASLAK') & (self.data['WEEKEND'] == 1),
                                              'MASLAK_WE', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'TAKSIM') & (self.data['WEEKEND'] == 0),
                                              'TAKSIM_WD', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'TAKSIM') & (self.data['WEEKEND'] == 1),
                                              'TAKSIM_WE', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'HISTORIC') & (self.data['WEEKEND'] == 0),
                                              'HISTORIC_WD', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'HISTORIC') & (self.data['WEEKEND'] == 1),
                                              'HISTORIC_WE', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'LEVENT') & (self.data['WEEKEND'] == 0),
                                              'LEVENT_WD', self.data['CASESTUDY'])
            self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'LEVENT') & (self.data['WEEKEND'] == 1),
                                              'LEVENT_WE', self.data['CASESTUDY'])
            '''self.data.replace(
                to_replace=['UNIVERSITY_WE', 'UNIVERSITY_WD', 'MASLAK_WE', 'MASLAK_WD', 'TAKSIM_WE', 'TAKSIM_WD',
                            'HISTORIC_WE', 'HISTORIC_WD', 'LEVENT_WE', 'LEVENT_WD'],
                value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                inplace=True)'''
            self.x = self.data.values
            self.y = self.data['CASESTUDY'].values
            self.x = np.delete(self.x, 0, axis=1)
            self.x = np.delete(self.x, 0, axis=1)
            self.x = np.delete(self.x, 0, axis=1)

	#funzione per plottare le time series medie normalizzate
    def plotTest(self):
        self.data = self.data.drop(columns=['MONTH', 'DAY', 'WEEKEND'])
        self.data = self.data.groupby('CASESTUDY').mean('media')
        self.data = self.data.T
        x = self.data
        # minMaxScaler per normalizzare
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns=x.columns)
        df = df.rename(
            index={0: 'H0', 1: 'H1', 2: 'H2', 3: 'H3', 4: 'H4', 5: 'H5', 6: 'H6', 7: 'H7', 8: 'H8', 9: 'H9', 10: 'H10',
                   11: 'H11', 12: 'H12', 13: 'H13', 14: 'H14', 15: 'H15', 16: 'H16', 17: 'H17', 18: 'H18', 19: 'H19',
                   20: 'H20', 21: 'H21', 22: 'H22', 23: 'H23', 24: 'H24'})
        self.data = df.T
        plotta = df[['MASLAK_WD', 'UNIVERSITY_WD','LEVENT_WD','HISTORIC_WD','TAKSIM_WD']]
        plotta.plot(marker='o', linewidth=2, markersize=5)
        plt.savefig('plotTimeSeriesMedioGeneraleWD.png')

	#funzione di normalizzazione
    def NormalizedData(self):
        x = self.x
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.x = x_scaled

	#plot casuale di una time series, sia normalizzata che non
    def righeCasuali(self):
        self.data.replace(to_replace=['MASLAK - INDUSTRIES', 'TAKSIM - SHOPPING', 'HISTORIC - TOURISM'],
                          value=['MASLAK', 'TAKSIM', 'HISTORIC'], inplace=True)
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'UNIVERSITY') & (self.data['WEEKEND'] == 0),
                                          'UNIVERSITY_WD', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'UNIVERSITY') & (self.data['WEEKEND'] == 1),
                                          'UNIVERSITY_WE', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'MASLAK') & (self.data['WEEKEND'] == 0),
                                          'MASLAK_WD', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'MASLAK') & (self.data['WEEKEND'] == 1),
                                          'MASLAK_WE', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'TAKSIM') & (self.data['WEEKEND'] == 0),
                                          'TAKSIM_WD', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'TAKSIM') & (self.data['WEEKEND'] == 1),
                                          'TAKSIM_WE', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'HISTORIC') & (self.data['WEEKEND'] == 0),
                                          'HISTORIC_WD', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'HISTORIC') & (self.data['WEEKEND'] == 1),
                                          'HISTORIC_WE', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'LEVENT') & (self.data['WEEKEND'] == 0),
                                          'LEVENT_WD', self.data['CASESTUDY'])
        self.data['CASESTUDY'] = np.where((self.data['CASESTUDY'] == 'LEVENT') & (self.data['WEEKEND'] == 1),
                                          'LEVENT_WE', self.data['CASESTUDY'])
        self.data = self.data.drop(columns=['MONTH', 'DAY', 'WEEKEND'])
        # seleziono un indice casuale
        dim = len(self.data.index)
        import random
        indice = random.randint(0, dim)
        df = self.data.iloc[[indice], :]
        df = pd.DataFrame(df)

        df_normale = df
        df_normale = df_normale.groupby('CASESTUDY').mean()
        df_normale = df_normale.T

        df_normalizzato = df.groupby('CASESTUDY').mean()
        df_normalizzato = df_normalizzato.T
        x = df_normalizzato
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        plotta = pd.DataFrame(x_scaled, columns=x.columns)
        plotta = plotta.rename(
            index={0: 'H0', 1: 'H1', 2: 'H2', 3: 'H3', 4: 'H4', 5: 'H5', 6: 'H6', 7: 'H7', 8: 'H8', 9: 'H9', 10: 'H10',
                   11: 'H11', 12: 'H12', 13: 'H13', 14: 'H14', 15: 'H15', 16: 'H16', 17: 'H17', 18: 'H18', 19: 'H19',
                   20: 'H20', 21: 'H21', 22: 'H22', 23: 'H23', 24: 'H24'})
        plotta.plot(marker='o', linewidth=2, markersize=5)
        plt.savefig('plot_confronta_1.png')
        df_normale.plot(marker='o', linewidth=2, markersize=5)
        plt.savefig('plot_confronta_2.png')

	#random forest
    def trainForest(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=self.var_test_size,
                                                                                random_state=0)
        self.trained_model = ensemble.RandomForestClassifier(n_jobs=-1, random_state=self.var_random_state,
                                                             n_estimators=self.var_n_estimators) \
            .fit(self.x_train, self.y_train)

	#score e plot della confusion matrix
    def evoluationCase1(self):
        # calcolo lo score e lo riporto in un file
        score = self.trained_model.score(self.x_test, self.y_test)
        f = open("report.txt", "a")
        #f = open("task2/report.txt", "a")
        f.write('Test score: ' + str(score) + ' test_size ' + str(self.var_test_size) + ' estimators ' + str(
            self.var_n_estimators) + ' random_state ' + str(self.var_random_state) + '\n')
        # la matrice di confusione viene salvata in forma grafica
        plot_confusion_matrix(self.trained_model, self.x_test, self.y_test)
        plt.savefig('report_' + str(score) + '.png')
        #plt.savefig('task2/report_' + str(score) + '.png')
        return score

